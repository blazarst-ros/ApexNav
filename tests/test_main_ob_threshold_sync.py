from pathlib import Path


def test_eval_detector_thresholds_match_main_ob():
    expected = {
        Path("config/habitat_eval_hm3dv1.yaml"): (
            "confidence_threshold_yolo: 0.20",
            "confidence_threshold_dino: 0.40",
        ),
        Path("config/habitat_eval_hm3dv2.yaml"): (
            "confidence_threshold_yolo: 0.20",
            "confidence_threshold_dino: 0.30",
        ),
        Path("config/habitat_eval_mp3d.yaml"): (
            "confidence_threshold_yolo: 0.20",
            "confidence_threshold_dino: 0.30",
        ),
    }

    for config_path, values in expected.items():
        text = config_path.read_text(encoding="utf-8")
        for value in values:
            assert value in text


def test_object_map_dbscan_matches_main_ob():
    text = Path("src/planner/plan_env/src/map_ros.cpp").read_text(encoding="utf-8")
    assert "dbscan(single_object_cloud, 0.15f, 6)" in text
