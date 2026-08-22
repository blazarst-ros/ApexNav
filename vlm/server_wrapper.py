import base64
import os
import threading
import time
from typing import Any, Dict

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, request


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


def send_request(url: str, **kwargs: Any) -> dict:
    """Match Lite-ApexNav's bounded request and retry contract."""
    max_attempts = int(
        kwargs.pop("max_attempts", os.environ.get("VLM_REQUEST_ATTEMPTS", 1))
    )
    retry_backoff = float(
        kwargs.pop("retry_backoff", os.environ.get("VLM_RETRY_BACKOFF", 0.5))
    )
    last_error = None
    for attempt in range(max(1, max_attempts)):
        try:
            return _send_request(url, **kwargs)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max_attempts:
                time.sleep(retry_backoff * (attempt + 1))
    raise RuntimeError(
        f"VLM request to {url} failed after {max(1, max_attempts)} attempts: {last_error}"
    ) from last_error


def _send_request(url: str, **kwargs: Any) -> dict:
    """Perform one Lite-ApexNav HTTP request window after payload encoding."""
    request_timeout = float(
        kwargs.pop("request_timeout", os.environ.get("VLM_REQUEST_TIMEOUT", 5))
    )
    payload = {
        key: image_to_str(value, quality=kwargs.get("quality", 90))
        if isinstance(value, np.ndarray)
        else value
        for key, value in kwargs.items()
    }
    deadline = time.monotonic() + request_timeout
    while True:
        remaining_timeout = deadline - time.monotonic()
        if remaining_timeout <= 0:
            raise Exception(f"Request timed out after {request_timeout} seconds")
        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=remaining_timeout,
            )
            if response.status_code == 200:
                return response.json()
            raise RuntimeError(
                f"Request failed with HTTP {response.status_code}: {response.text[:200]}"
            )
        except (requests.exceptions.Timeout, requests.exceptions.RequestException):
            if time.monotonic() >= deadline:
                raise Exception(f"Request timed out after {request_timeout} seconds")
            time.sleep(min(1.0, deadline - time.monotonic()))
