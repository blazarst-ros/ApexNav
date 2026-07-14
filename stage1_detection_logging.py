import json
import math
import os
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

import numpy as np


DEFAULT_STAGE1_DETECTION_DIR = "/media/blazarst/Getea/RuntimeData/Stage1_detector"

STAGE1_DETECTION_SCHEMA = {
    "schema_name": "stage1_detection",
    "schema_version": 1,
    "format": "jsonl",
    "fields": {
        "episode_id": {"type": "string", "required": True},
        "step_index": {"type": "uint32", "required": True},
        "detection_id": {"type": "uint32", "required": True},
        "robot_id": {"type": "uint8", "required": True},
        "target_label": {"type": "string", "required": True},
        "top1_label": {"type": "string", "required": True},
        "top1_score": {"type": "float32", "required": True},
        "top2_label": {"type": "string", "required": True},
        "top2_score": {"type": "float32", "required": True},
        "camera_height": {
            "type": "float32",
            "unit": "meter",
            "required": True,
        },
        "object_distance": {
            "type": "float32",
            "unit": "meter",
            "required": True,
        },
        "mask_area_ratio": {"type": "float32", "required": True},
        "height_utility": {"type": "float32", "required": True},
        "final_utility": {"type": "float32", "required": True},
        "stamp_sec": {
            "type": "float64",
            "unit": "second",
            "required": True,
        },
    },
}


def ensure_stage1_detection_storage(output_dir) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    schema_path = output_path / "schema_stage1_detection.json"
    if not schema_path.exists():
        schema_path.write_text(
            json.dumps(STAGE1_DETECTION_SCHEMA, ensure_ascii=True, indent=2)
            + "\n",
            encoding="utf-8",
        )
    return output_path


def write_stage1_detection_records(output_dir, episode_id: str, records: Iterable[Mapping]):
    records = list(records)
    if not records:
        return
    output_path = ensure_stage1_detection_storage(output_dir)
    jsonl_path = output_path / f"{episode_id}.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(dict(record), ensure_ascii=True, sort_keys=True))
            file.write("\n")


def build_stage1_detection_records(
    *,
    episode_id: str,
    step_index: int,
    robot_id: int,
    target_label: str,
    score_list: Sequence,
    object_masks_list: Sequence,
    label_list: Sequence,
    depth,
    camera_height: float,
    confusion_labels: Sequence[str],
    stamp_sec: float,
) -> List[dict]:
    records = []
    for score, mask, label_index in zip(score_list, object_masks_list, label_list):
        top1_label = _label_from_index(target_label, confusion_labels, label_index)
        top1_score = _safe_float(score)
        mask_area_ratio = _mask_area_ratio(mask)
        object_distance = _object_distance(mask, depth)

        if not _is_valid_detection(
            top1_label=top1_label,
            top1_score=top1_score,
            mask_area_ratio=mask_area_ratio,
            object_distance=object_distance,
        ):
            continue

        records.append(
            {
                "episode_id": str(episode_id),
                "step_index": int(step_index),
                "detection_id": len(records),
                "robot_id": int(robot_id),
                "target_label": str(target_label),
                "top1_label": top1_label,
                "top1_score": top1_score,
                "top2_label": "",
                "top2_score": 0.0,
                "camera_height": _safe_float(camera_height),
                "object_distance": object_distance,
                "mask_area_ratio": mask_area_ratio,
                "height_utility": 0.0,
                "final_utility": top1_score,
                "stamp_sec": _safe_float(stamp_sec),
            }
        )
    return records


def _label_from_index(target_label: str, confusion_labels: Sequence[str], label_index) -> str:
    try:
        index = int(label_index)
    except (TypeError, ValueError):
        return ""
    if index == 0:
        return str(target_label)
    if 1 <= index <= len(confusion_labels):
        return str(confusion_labels[index - 1])
    return ""


def _mask_area_ratio(mask) -> float:
    mask_array = np.asarray(mask)
    if mask_array.size == 0:
        return 0.0
    return float(np.count_nonzero(mask_array) / mask_array.size)


def _object_distance(mask, depth) -> float:
    if depth is None:
        return float("inf")
    mask_array = np.asarray(mask).astype(bool)
    depth_array = np.asarray(depth)
    if depth_array.ndim == 3 and depth_array.shape[-1] == 1:
        depth_array = depth_array[:, :, 0]
    if mask_array.shape != depth_array.shape:
        return float("inf")
    values = depth_array[mask_array].astype(np.float64)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return float("inf")
    return float(np.median(values))


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _is_valid_detection(
    *,
    top1_label: str,
    top1_score: float,
    mask_area_ratio: float,
    object_distance: float,
) -> bool:
    return (
        top1_label != ""
        and mask_area_ratio >= 0.001
        and 0.1 <= object_distance <= 5.0
        and math.isfinite(top1_score)
    )


def get_stage1_detection_output_dir() -> str:
    return os.environ.get(
        "APEXNAV_STAGE1_DETECTOR_DIR",
        DEFAULT_STAGE1_DETECTION_DIR,
    )
