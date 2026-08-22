from pathlib import Path


def test_eval_detector_thresholds_use_the_lite_yoloe_contract():
    for config_path in (
        Path("config/habitat_eval_hm3dv1.yaml"),
        Path("config/habitat_eval_hm3dv2.yaml"),
        Path("config/habitat_eval_mp3d.yaml"),
    ):
        text = config_path.read_text(encoding="utf-8")
        assert "  yoloe:" in text
        assert "    confidence_threshold: 0.3" in text
        assert "groundingDINO:" not in text
        assert "confidence_threshold_yolo:" not in text


def test_object_map_dbscan_matches_main_ob():
    text = Path("src/planner/plan_env/src/map_ros.cpp").read_text(encoding="utf-8")
    assert "dbscan(single_object_cloud, 0.15f, 6)" in text
