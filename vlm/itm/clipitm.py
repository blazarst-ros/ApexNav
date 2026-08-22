"""CLIP-based image-text matching service and REST client."""

import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from vlm.server_wrapper import ServerMixin, host_model, send_request, str_to_image

try:
    import clip
except Exception:
    clip = None


def _default_clip_download_root() -> Optional[str]:
    prefix = Path(os.environ.get("CONDA_PREFIX", sys.prefix)).expanduser()
    cache = prefix.parent.parent / "model-cache" / "clip"
    return str(cache) if cache.is_dir() else None


class CLIPITM:
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None) -> None:
        if clip is None:
            raise RuntimeError("CLIP is required to host the CLIPITM server")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        download_root = os.environ.get("CLIP_DOWNLOAD_ROOT") or _default_clip_download_root()
        kwargs = {"download_root": download_root} if download_root else {}
        self.model, self.preprocess = clip.load(model_name, device=self.device, **kwargs)
        self.model.eval()

    def infer(self, image: np.ndarray, txt: str):
        preprocess_start = time.perf_counter()
        image_input = self.preprocess(Image.fromarray(image)).unsqueeze(0).to(self.device)
        text_input = clip.tokenize([txt]).to(self.device)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        inference_start = time.perf_counter()
        with torch.inference_mode():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        model_inference_ms = (time.perf_counter() - inference_start) * 1000.0

        postprocess_start = time.perf_counter()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        cosine = float(torch.matmul(image_features, text_features.T).item())
        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0
        return cosine, float((cosine + 1.0) / 2.0), {
            "preprocess_ms": preprocess_ms,
            "model_inference_ms": model_inference_ms,
            "postprocess_ms": postprocess_ms,
        }

    def cosine(self, image: np.ndarray, txt: str) -> float:
        return self.infer(image, txt)[0]

    def itm_score(self, image: np.ndarray, txt: str) -> float:
        return self.infer(image, txt)[1]


class CLIPITMClient:
    def __init__(self, port: int = 12182, route_name: str = "clipitm") -> None:
        self.url = f"http://localhost:{port}/{route_name}"

    def infer(self, image: np.ndarray, txt: str) -> dict:
        request_start = time.perf_counter()
        response = send_request(self.url, image=image, txt=txt)
        timing = dict(response.get("timing", {}))
        timing["client_total_ms"] = (time.perf_counter() - request_start) * 1000.0
        response["timing"] = timing
        return response

    def cosine(self, image: np.ndarray, txt: str) -> float:
        return float(self.infer(image, txt)["response"])

    def itm_score(self, image: np.ndarray, txt: str) -> float:
        return float(self.infer(image, txt)["itm score"])


def build_server(route_name: str = "clipitm"):
    class CLIPITMServer(ServerMixin, CLIPITM):
        def process_payload(self, payload: dict) -> dict:
            request_start = time.perf_counter()
            cosine, itm_score, timing = self.infer(str_to_image(payload["image"]), payload["txt"])
            timing["server_total_ms"] = (time.perf_counter() - request_start) * 1000.0
            return {"response": cosine, "itm score": itm_score, "timing": timing}

    return CLIPITMServer, route_name


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12182)
    parser.add_argument("--model", type=str, default="ViT-B/32")
    args = parser.parse_args()
    Server, route = build_server()
    host_model(Server(model_name=args.model), name=route, port=args.port)
