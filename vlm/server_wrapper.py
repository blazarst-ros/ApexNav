import base64
import os
import threading
import time
from typing import Any, Dict

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, request


VLM_REQUEST_TIMEOUT = float(os.environ.get("VLM_REQUEST_TIMEOUT", "5"))


class ServerMixin:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def process_payload(self, payload: dict) -> dict:
        raise NotImplementedError


def host_model(model: Any, name: str, port: int = 5000, run: bool = True) -> Flask:
    """Create a single-process, serialized REST endpoint for a VLM model."""
    app = Flask(__name__)
    request_lock = threading.Lock()

    @app.route("/healthz", methods=["GET"])
    def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.route(f"/{name}", methods=["POST"])
    def process_request() -> Dict[str, Any]:
        payload = request.get_json()
        with request_lock:
            response = model.process_payload(payload)
        return jsonify(response)

    if run:
        app.run(host="localhost", port=port, threaded=True)
    return app


def bool_arr_to_str(arr: np.ndarray) -> str:
    """Convert a uint8 mask array to a JSON-safe string."""
    return base64.b64encode(arr.tobytes()).decode()


def str_to_bool_arr(s: str, shape: tuple) -> np.ndarray:
    """Convert a JSON-safe uint8 mask string back to an array."""
    return np.frombuffer(base64.b64decode(s), dtype=np.uint8).reshape(shape)


def image_to_str(img_np: np.ndarray, quality: float = 90.0) -> str:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    _, buffer = cv2.imencode(".jpg", img_np, encode_param)
    return base64.b64encode(buffer).decode("utf-8")


def str_to_image(img_str: str) -> np.ndarray:
    img_bytes = base64.b64decode(img_str)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_arr, cv2.IMREAD_ANYCOLOR)


def send_request(
    url: str,
    *,
    max_attempts: int = 1,
    backoff_seconds: float = 0.0,
    timeout: float = VLM_REQUEST_TIMEOUT,
    **kwargs: Any,
) -> dict:
    """Send a bounded VLM request without waiting for stale work to drain."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")
    if backoff_seconds < 0:
        raise ValueError("backoff_seconds cannot be negative")
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    deadline = time.monotonic() + timeout
    for attempt in range(max_attempts):
        try:
            return _send_request(url, deadline=deadline, **kwargs)
        except requests.exceptions.RequestException:
            if attempt == max_attempts - 1 or time.monotonic() >= deadline:
                raise
            sleep_seconds = min(backoff_seconds, max(0.0, deadline - time.monotonic()))
            if sleep_seconds:
                time.sleep(sleep_seconds)
    raise RuntimeError("unreachable")


def _send_request(url: str, *, deadline: float, **kwargs: Any) -> dict:
    """Perform one HTTP request using the caller's end-to-end deadline."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise requests.exceptions.Timeout("VLM request deadline expired")

    payload = {
        key: image_to_str(value, quality=kwargs.get("quality", 90))
        if isinstance(value, np.ndarray)
        else value
        for key, value in kwargs.items()
    }
    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=remaining,
    )
    if response.status_code != 200:
        response.raise_for_status()
    return response.json()
