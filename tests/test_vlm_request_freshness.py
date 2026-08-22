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
                with self.assertRaises(server_wrapper.requests.exceptions.ConnectionError):
                    server_wrapper.send_request("http://localhost:12184/yoloe")

        self.assertEqual(send.call_count, 1)
        sleep.assert_not_called()

    def test_request_retries_only_when_attempts_and_backoff_are_explicit(self):
        """Catches retry/backoff that cannot be bounded by the caller."""
        with patch.object(
            server_wrapper,
            "_send_request",
            side_effect=[server_wrapper.requests.exceptions.ConnectionError("transient"), {"response": 0.5}],
        ) as send:
            with patch.object(server_wrapper.time, "sleep") as sleep:
                result = server_wrapper.send_request(
                    "http://localhost:12182/clipitm",
                    max_attempts=2,
                    backoff_seconds=0.1,
                    image=np.zeros((2, 2, 3), dtype=np.uint8),
                )

        self.assertEqual(result, {"response": 0.5})
        self.assertEqual(send.call_count, 2)
        sleep.assert_called_once_with(0.1)

    def test_send_uses_a_total_deadline_without_lockfiles(self):
        """Catches cross-process lock waiting instead of an end-to-end request deadline."""
        response = Mock(status_code=200)
        response.json.return_value = {"response": 0.5}
        with patch.object(server_wrapper.time, "monotonic", return_value=1.5):
            with patch.object(server_wrapper.requests, "post", return_value=response) as post:
                result = server_wrapper._send_request(
                    "http://localhost:12182/clipitm",
                    deadline=5.0,
                    image=np.zeros((2, 2, 3), dtype=np.uint8),
                )

        self.assertEqual(result, {"response": 0.5})
        self.assertEqual(post.call_args.kwargs["timeout"], 3.5)
        source = Path("vlm/server_wrapper.py").read_text(encoding="utf-8")
        self.assertNotIn("lockfiles", source)
        self.assertNotIn("import random", source)
        self.assertNotIn("import socket", source)

    def test_host_model_exposes_healthz_and_serializes_model_requests(self):
        """Catches a model server accepting concurrent inference without a process lock."""
        model = Mock()
        model.process_payload.return_value = {"response": 0.5}
        app = server_wrapper.host_model(model, "clipitm", run=False)

        client = app.test_client()
        health = client.get("/healthz")
        inference = client.post("/clipitm", json={"txt": "chair"})

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.get_json(), {"status": "ok"})
        self.assertEqual(inference.get_json(), {"response": 0.5})
        source = Path("vlm/server_wrapper.py").read_text(encoding="utf-8")
        self.assertIn("request_lock = threading.Lock()", source)
        self.assertIn("with request_lock:", source)

    def test_multiagent_entrypoints_use_current_agent_observation_without_image_backlog(self):
        """Catches automatic or manual VLM calls being moved behind an image queue."""
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
