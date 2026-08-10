from pathlib import Path


MAP_ROS_SOURCE = Path("src/planner/plan_env/src/map_ros.cpp")
README = Path("README.md")
GROUNDING_DINO_CONFIGS = [
    Path("config/habitat_eval_mp3d.yaml"),
    Path("config/habitat_eval_hm3dv2.yaml"),
    Path("config/habitat_vel_control.yaml"),
]
YOLO_CONFIGS = [
    Path("config/habitat_eval_mp3d.yaml"),
    Path("config/habitat_eval_hm3dv2.yaml"),
    Path("config/habitat_eval_hm3dv1.yaml"),
    Path("config/habitat_vel_control.yaml"),
    Path("real_world_test_example/config/real_world_test.yaml"),
]


def test_over_depth_object_cloud_is_kept_for_15_callbacks():
    source = MAP_ROS_SOURCE.read_text(encoding="utf-8")

    assert "continue_over_depth_count_ <= 15" in source
    assert "continue_over_depth_count_ <= 4" not in source


def test_current_over_depth_detection_refreshes_cache_before_reusing_last_cloud():
    source = MAP_ROS_SOURCE.read_text(encoding="utf-8")

    assert "bool has_current_over_depth_object_cloud" in source
    assert source.index("bool has_current_over_depth_object_cloud") < source.index(
        "*map_->object_map2d_->over_depth_object_cloud_ = *last_over_depth_cloud"
    )
    assert "if (has_current_over_depth_object_cloud)" in source
    assert "continue_over_depth_count_ = 0;" in source


def test_readme_documents_over_depth_cache_change():
    readme = README.read_text(encoding="utf-8")

    assert "over_depth_object_cloud_" in readme
    assert "15" in readme
    assert "SEARCH_OVER_DEPTH_OBJECT" in readme


def test_object_dbscan_uses_more_permissive_cluster_settings():
    source = MAP_ROS_SOURCE.read_text(encoding="utf-8")

    assert "dbscan(single_object_cloud, 0.15f, 6)" in source
    assert "dbscan(single_object_cloud, 0.12f, 10)" not in source


def test_grounding_dino_thresholds_are_relaxed_for_sparse_object_clouds():
    for config_path in GROUNDING_DINO_CONFIGS:
        config = config_path.read_text(encoding="utf-8")

        assert "confidence_threshold_dino: 0.30" in config
        assert "text_threshold: 0.20" in config


def test_yolo_confidence_threshold_is_relaxed_for_sparse_object_clouds():
    for config_path in YOLO_CONFIGS:
        config = config_path.read_text(encoding="utf-8")

        assert "confidence_threshold_yolo: 0.20" in config
        assert "confidence_threshold_yolo: 0.25" not in config
        assert "confidence_threshold_yolo: 0.3" not in config


def test_readme_documents_detection_threshold_tuning():
    readme = README.read_text(encoding="utf-8")

    assert "GroundingDINO" in readme
    assert "YOLO" in readme
    assert "confidence_threshold_yolo: 0.20" in readme
    assert "confidence_threshold_dino: 0.30" in readme
    assert "text_threshold: 0.20" in readme
    assert "dbscan(single_object_cloud, 0.15f, 6)" in readme
