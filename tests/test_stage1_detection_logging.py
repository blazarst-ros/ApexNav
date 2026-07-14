import json

import numpy as np

from stage1_detection_logging import (
    DEFAULT_STAGE1_DETECTION_DIR,
    build_stage1_detection_records,
    ensure_stage1_detection_storage,
    write_stage1_detection_records,
)


def test_default_stage1_detection_dir_uses_getea_runtime_path():
    assert (
        DEFAULT_STAGE1_DETECTION_DIR
        == "/media/blazarst/Getea/RuntimeData/Stage1_detector"
    )


def test_build_records_filters_invalid_detections_and_assigns_ids():
    depth = np.full((4, 4), 1.5, dtype=np.float32)
    depth[2:, :] = 6.0
    valid_mask = np.zeros((4, 4), dtype=np.uint8)
    valid_mask[:2, :2] = 1
    empty_mask = np.zeros((4, 4), dtype=np.uint8)
    far_mask = np.zeros((4, 4), dtype=np.uint8)
    far_mask[2:, :] = 1

    records = build_stage1_detection_records(
        episode_id="ep-1",
        step_index=7,
        robot_id=0,
        target_label="chair",
        score_list=[0.8, float("nan"), 0.7, 0.6],
        object_masks_list=[valid_mask, valid_mask, empty_mask, far_mask],
        label_list=[0, 1, 0, 0],
        depth=depth,
        camera_height=0.9,
        confusion_labels=["sofa"],
        stamp_sec=12.25,
    )

    assert len(records) == 1
    assert records[0]["detection_id"] == 0
    assert records[0]["episode_id"] == "ep-1"
    assert records[0]["step_index"] == 7
    assert records[0]["robot_id"] == 0
    assert records[0]["target_label"] == "chair"
    assert records[0]["top1_label"] == "chair"
    assert records[0]["top1_score"] == 0.8
    assert records[0]["top2_label"] == ""
    assert records[0]["top2_score"] == 0.0
    assert records[0]["camera_height"] == 0.9
    assert records[0]["object_distance"] == 1.5
    assert records[0]["mask_area_ratio"] == 0.25
    assert records[0]["height_utility"] == 0.0
    assert records[0]["final_utility"] == 0.8
    assert records[0]["stamp_sec"] == 12.25


def test_write_records_creates_schema_and_episode_jsonl(tmp_path):
    output_dir = tmp_path / "Stage1_detector"
    ensure_stage1_detection_storage(output_dir)

    schema_path = output_dir / "schema_stage1_detection.json"
    assert schema_path.exists()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["schema_name"] == "stage1_detection"
    assert "object_distance" in schema["fields"]

    records = [
        {
            "episode_id": "ep-2",
            "step_index": 1,
            "detection_id": 0,
            "robot_id": 0,
            "target_label": "chair",
            "top1_label": "chair",
            "top1_score": 0.9,
            "top2_label": "",
            "top2_score": 0.0,
            "camera_height": 0.9,
            "object_distance": 1.0,
            "mask_area_ratio": 0.1,
            "height_utility": 0.0,
            "final_utility": 0.9,
            "stamp_sec": 100.0,
        }
    ]

    write_stage1_detection_records(output_dir, "ep-2", records)

    jsonl_path = output_dir / "ep-2.jsonl"
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == records[0]
