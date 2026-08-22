"""YOLOE segmentation server and lightweight REST client."""

import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch

from basic_utils.path_utils import PROJECT_ROOT, WORKSPACE_ROOT, resolve_existing_path
from vlm.detector.detections import ObjectDetections
from vlm.server_wrapper import (
    ServerMixin,
    bool_arr_to_str,
    host_model,
    send_request,
    str_to_bool_arr,
    str_to_image,
)

try:
    from ultralytics import YOLOE
except Exception:
    YOLOE = None


YOLOE_WEIGHTS = "data/yoloe-11l-seg.pt"


def _conda_model_cache() -> Path:
    prefix = Path(os.environ.get("CONDA_PREFIX", Path(sys.prefix))).expanduser()
    return prefix.parent.parent / "model-cache" / "yoloe-11l-seg.pt"


def _configure_ultralytics_asset_cache() -> None:
    asset_cache = _conda_model_cache().parent / "ultralytics"
    if not asset_cache.is_dir():
        return
    try:
        from ultralytics.utils import SETTINGS

        SETTINGS["weights_dir"] = str(asset_cache)
    except Exception:
        pass


@contextmanager
def _ultralytics_asset_workdir():
    asset_cache = _conda_model_cache().parent / "ultralytics"
    asset = asset_cache / "mobileclip_blt.ts"
    if not asset.is_file():
        yield
        return
    previous = Path.cwd()
    try:
        os.chdir(asset_cache)
        yield
    finally:
        os.chdir(previous)


class YOLOEDetector:
    def __init__(
        self,
        weights: str = YOLOE_WEIGHTS,
        image_size: int = 640,
        device: Optional[str] = None,
    ) -> None:
        if YOLOE is None:
            raise RuntimeError("ultralytics YOLOE is required to host the YOLOE server")
        weights = resolve_existing_path(
            weights,
            WORKSPACE_ROOT / YOLOE_WEIGHTS,
            PROJECT_ROOT / YOLOE_WEIGHTS,
            PROJECT_ROOT / "model-cache/yoloe-11l-seg.pt",
            WORKSPACE_ROOT / "Lite-ApexNav/model-cache/yoloe-11l-seg.pt",
            _conda_model_cache(),
        )
        self.image_size = image_size
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        _configure_ultralytics_asset_cache()
        self.model = YOLOE(weights)
        self._predict_lock = threading.Lock()

    def warmup(self) -> None:
        with _ultralytics_asset_workdir():
            self.predict(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8), ["chair"])

    def predict(
        self,
        image: np.ndarray,
        classes: Sequence[str],
        agnostic_nms: bool = True,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ) -> ObjectDetections:
        labels = list(dict.fromkeys(label.strip() for label in classes if label and label.strip()))
        if not labels:
            return self._empty_detections(image)
        with self._predict_lock, torch.inference_mode():
            self.model.set_classes(labels)
            result = self.model.predict(
                source=image,
                conf=conf_thres,
                iou=iou_thres,
                agnostic_nms=agnostic_nms,
                imgsz=self.image_size,
                device=self.device,
                verbose=False,
            )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return self._empty_detections(image)
        boxes = result.boxes.xyxyn.detach().cpu().float()
        logits = result.boxes.conf.detach().cpu().float()
        phrases = [result.names[int(index)] for index in result.boxes.cls.detach().cpu().int().tolist()]
        return ObjectDetections(
            boxes=boxes,
            logits=logits,
            phrases=phrases,
            masks=self._extract_masks(result, image.shape[:2]),
            image_source=image,
            fmt="xyxy",
        )

    @staticmethod
    def _empty_detections(image: np.ndarray) -> ObjectDetections:
        return ObjectDetections(
            boxes=torch.empty((0, 4), dtype=torch.float32),
            logits=torch.empty((0,), dtype=torch.float32),
            phrases=[],
            masks=[],
            image_source=image,
            fmt="xyxy",
        )

    @staticmethod
    def _extract_masks(result, image_shape):
        if result.masks is None or result.masks.data is None:
            return [None] * len(result.boxes)
        height, width = image_shape
        masks = []
        for mask in result.masks.data.detach().cpu().numpy():
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            masks.append((mask > 0.5).astype(np.uint8))
        return masks


class YOLOEClient:
    def __init__(self, port: int = 12184) -> None:
        self.url = f"http://localhost:{port}/yoloe"

    def predict(
        self,
        image_numpy: np.ndarray,
        classes: List[str],
        agnostic_nms: bool = True,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ) -> ObjectDetections:
        response = send_request(
            self.url,
            image=image_numpy,
            classes=classes,
            agnostic_nms=agnostic_nms,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
        )
        masks = [
            None if mask is None else str_to_bool_arr(mask, shape=tuple(image_numpy.shape[:2]))
            for mask in response.get("masks", [])
        ]
        masks.extend([None] * (len(response["phrases"]) - len(masks)))
        return ObjectDetections(
            boxes=torch.tensor(response["boxes"], dtype=torch.float32).reshape(-1, 4),
            logits=torch.tensor(response["logits"], dtype=torch.float32),
            phrases=response["phrases"],
            masks=masks,
            image_source=image_numpy,
            fmt="xyxy",
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12184)
    parser.add_argument("--weights", type=str, default=os.environ.get("YOLOE_WEIGHTS", YOLOE_WEIGHTS))
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    class YOLOEServer(ServerMixin, YOLOEDetector):
        def process_payload(self, payload: dict) -> dict:
            detections = self.predict(
                str_to_image(payload["image"]),
                classes=payload["classes"],
                agnostic_nms=payload["agnostic_nms"],
                conf_thres=payload["conf_thres"],
                iou_thres=payload["iou_thres"],
            )
            response = detections.to_json()
            response["masks"] = [
                None if mask is None else bool_arr_to_str(mask.astype(np.uint8))
                for mask in detections.masks
            ]
            return response

    server = YOLOEServer(weights=args.weights, image_size=args.imgsz)
    server.warmup()
    host_model(server, name="yoloe", port=args.port)
