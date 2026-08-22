import os
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from vlm import server_wrapper


class VLMRequestFreshnessTests(unittest.TestCase):
    def test_default_request_attempts_once_and_raises_without_sleeping(self):
        """Catches stale-frame retries caused by an unbounded default retry policy."""
        transient_error = server_wrapper.requests.exceptions.ConnectionError("down")
        with patch.object(server_wrapper, "_send_request", side_effect=transient_error) as send:
            with patch.object(server_wrapper.time, "sleep") as sleep:
                with self.assertRaisesRegex(RuntimeError, "failed after 1 attempts"):
                    server_wrapper.send_request("http://localhost:12184/yoloe")

        self.assertEqual(send.call_count, 1)
        sleep.assert_not_called()

    def test_request_honors_lite_environment_retry_settings(self):
        """Catches deployment env knobs ignored by a wrapper-specific retry API."""
        with patch.object(
            server_wrapper,
            "_send_request",
            side_effect=[Exception("transient"), {"response": 0.5}],
        ) as send:
            with patch.object(server_wrapper.time, "sleep") as sleep:
                with patch.dict(
                    os.environ,
                    {
                        "VLM_REQUEST_ATTEMPTS": "2",
                        "VLM_RETRY_BACKOFF": "0.25",
                        "VLM_REQUEST_TIMEOUT": "7",
                    },
                ):
                    result = server_wrapper.send_request(
                        "http://localhost:12182/clipitm",
                        image=np.zeros((2, 2, 3), dtype=np.uint8),
                    )

        self.assertEqual(result, {"response": 0.5})
        self.assertEqual(send.call_count, 2)
        sleep.assert_called_once_with(0.25)
        self.assertNotIn("request_timeout", send.call_args.kwargs)

    def test_request_honors_lite_keyword_retry_settings(self):
        """Catches renamed retry keywords breaking existing Lite launch wrappers."""
        with patch.object(
            server_wrapper,
            "_send_request",
            side_effect=[Exception("transient"), {"response": 0.5}],
        ) as send:
            with patch.object(server_wrapper.time, "sleep") as sleep:
                result = server_wrapper.send_request(
                    "http://localhost:12182/clipitm",
                    max_attempts=2,
                    retry_backoff=0.125,
                    request_timeout=3,
                )

        self.assertEqual(result, {"response": 0.5})
        self.assertEqual(send.call_count, 2)
        sleep.assert_called_once_with(0.125)
        self.assertEqual(send.call_args.kwargs["request_timeout"], 3)

    def test_http_deadline_starts_after_image_encoding_without_lockfiles(self):
        """Catches a deadline established before the Lite payload encoding phase."""
        response = Mock(status_code=200)
        response.json.return_value = {"response": 0.5}
        encoded = threading.Event()

        def encode_image(*args, **kwargs):
            encoded.set()
            return "encoded-image"

        def monotonic_after_encoding():
            self.assertTrue(encoded.is_set())
            return 10.0

        with patch.object(server_wrapper, "image_to_str", side_effect=encode_image):
            with patch.object(server_wrapper.time, "monotonic", side_effect=monotonic_after_encoding):
                with patch.object(server_wrapper.requests, "post", return_value=response) as post:
                    with patch.dict(os.environ, {"VLM_REQUEST_TIMEOUT": "5"}):
                        result = server_wrapper._send_request(
                            "http://localhost:12182/clipitm",
                            image=np.zeros((2, 2, 3), dtype=np.uint8),
                        )

        self.assertEqual(result, {"response": 0.5})
        self.assertEqual(post.call_args.kwargs["timeout"], 5.0)
        source = Path("vlm/server_wrapper.py").read_text(encoding="utf-8")
        self.assertNotIn("lockfiles", source)
        self.assertNotIn("import random", source)
        self.assertNotIn("import socket", source)

    def test_host_model_exposes_healthz_and_serializes_concurrent_model_requests(self):
        """Catches a model server accepting concurrent inference without a process lock."""
        entered = threading.Event()
        release = threading.Event()
        state_lock = threading.Lock()
        active = 0
        max_active = 0

        class BlockingModel:
            def process_payload(self, payload):
                nonlocal active, max_active
                with state_lock:
                    active += 1
                    max_active = max(max_active, active)
                    entered.set()
                release.wait(timeout=1)
                with state_lock:
                    active -= 1
                return {"response": payload["response"]}

        model = BlockingModel()
        app = server_wrapper.host_model(model, "clipitm", run=False)

        client = app.test_client()
        health = client.get("/healthz")
        responses = []

        def invoke(value):
            with app.test_client() as thread_client:
                responses.append(thread_client.post("/clipitm", json={"response": value}))

        first = threading.Thread(target=invoke, args=(0.5,))
        second = threading.Thread(target=invoke, args=(0.6,))
        first.start()
        self.assertTrue(entered.wait(timeout=1))
        second.start()
        time.sleep(0.05)
        with state_lock:
            self.assertEqual(max_active, 1)
        release.set()
        first.join(timeout=1)
        second.join(timeout=1)

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.get_json(), {"status": "ok"})
        self.assertEqual(sorted(response.get_json()["response"] for response in responses), [0.5, 0.6])

    def test_multiagent_entrypoints_call_vlm_inline_from_current_observation(self):
        """Catches an entrypoint adding its own pending image-work queue."""
        automatic = Path("habitat_evaluation.py").read_text(encoding="utf-8")
        manual = Path("habitat_manual_control_multiagent.py").read_text(encoding="utf-8")

        self.assertIn('get_itm_message_cosine(img_np, label, room)', automatic)
        self.assertIn('get_object(\n                        label, img_np, detector_cfg, llm_answer', automatic)
        self.assertIn('get_itm_message_cosine(agent_obs["rgb"], label, room)', manual)
        self.assertIn('label, agent_obs["rgb"], detector_cfg, llm_answer', manual)
        for source in (automatic, manual):
            self.assertNotIn("pending", source.lower())
            self.assertNotIn("deque(", source)
            self.assertNotIn("Queue(", source)


if __name__ == "__main__":
    unittest.main()
