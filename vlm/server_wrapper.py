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


def host_model(model: Any, name: str, port: int = 5000) -> None:
    """
    Hosts a model as a REST API using Flask.
    """
    app = Flask(__name__)
    request_lock = threading.Lock()
    started_at = time.time()

    @app.route("/healthz", methods=["GET"])
    @app.route(f"/{name}/healthz", methods=["GET"])
    def health() -> Dict[str, Any]:
        return jsonify(
            ready=True,
            name=name,
            model_type=type(model).__name__,
            uptime_sec=time.time() - started_at,
        )

    @app.route(f"/{name}", methods=["POST"])
    def process_request() -> Dict[str, Any]:
        payload = request.json
        with request_lock:
            return jsonify(model.process_payload(payload))

    app.run(host="localhost", port=port)


def bool_arr_to_str(arr: np.ndarray) -> str:
    """Converts a boolean array to a string."""
    packed_str = base64.b64encode(arr.tobytes()).decode()
    return packed_str


def str_to_bool_arr(s: str, shape: tuple) -> np.ndarray:
    """Converts a string to a boolean array."""
    # Convert the string back into bytes using base64 decoding
    bytes_ = base64.b64decode(s)

    # Convert bytes to np.uint8 array
    bytes_array = np.frombuffer(bytes_, dtype=np.uint8)

    # Reshape the data back into a boolean array
    unpacked = bytes_array.reshape(shape)
    return unpacked


def image_to_str(img_np: np.ndarray, quality: float = 90.0) -> str:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    retval, buffer = cv2.imencode(".jpg", img_np, encode_param)
    img_str = base64.b64encode(buffer).decode("utf-8")
    return img_str


def str_to_image(img_str: str) -> np.ndarray:
    img_bytes = base64.b64decode(img_str)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img_np = cv2.imdecode(img_arr, cv2.IMREAD_ANYCOLOR)
    return img_np


def send_request(url: str, **kwargs: Any) -> dict:
    """Send a bounded request and surface failure to the ROS caller.

    Older code retried for several minutes and then called ``exit()``, which
    could kill a navigation process and allowed an old inference to arrive long
    after its source frame.  Callers now own the recovery policy.
    """
    max_attempts = int(kwargs.pop("max_attempts", os.environ.get("VLM_REQUEST_ATTEMPTS", 1)))
    retry_backoff = float(kwargs.pop("retry_backoff", os.environ.get("VLM_RETRY_BACKOFF", 0.5)))
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
    request_timeout = float(
        kwargs.pop("request_timeout", os.environ.get("VLM_REQUEST_TIMEOUT", 5))
    )

    # Create a payload dict which is a clone of kwargs but all np.array values are
    # converted to strings
    payload = {}
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            payload[k] = image_to_str(v, quality=kwargs.get("quality", 90))
        else:
            payload[k] = v
    # Set the headers
    headers = {"Content-Type": "application/json"}

    deadline = time.monotonic() + request_timeout
    while True:
        remaining_timeout = deadline - time.monotonic()
        if remaining_timeout <= 0:
            raise Exception(f"Request timed out after {request_timeout} seconds")

        try:
            resp = requests.post(
                url, headers=headers, json=payload, timeout=remaining_timeout
            )
            if resp.status_code == 200:
                result = resp.json()
                break
            else:
                raise RuntimeError(f"Request failed with HTTP {resp.status_code}: {resp.text[:200]}")
        except (
            requests.exceptions.Timeout,
            requests.exceptions.RequestException,
        ) as e:
            print(e)
            if time.monotonic() >= deadline:
                raise Exception(f"Request timed out after {request_timeout} seconds")
            time.sleep(min(1.0, deadline - time.monotonic()))

    return result
