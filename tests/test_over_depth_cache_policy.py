from pathlib import Path


MAP_ROS_SOURCE = Path("src/planner/plan_env/src/map_ros.cpp")
README = Path("README.md")


def test_over_depth_object_cloud_is_kept_for_15_callbacks():
    source = MAP_ROS_SOURCE.read_text(encoding="utf-8")

    assert "continue_over_depth_count_ <= 15" in source
    assert "continue_over_depth_count_ <= 4" not in source


def test_readme_documents_over_depth_cache_change():
    readme = README.read_text(encoding="utf-8")

    assert "over_depth_object_cloud_" in readme
    assert "15" in readme
    assert "SEARCH_OVER_DEPTH_OBJECT" in readme
