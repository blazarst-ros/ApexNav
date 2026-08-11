from pathlib import Path


MAP_ROS_CPP = Path("src/planner/plan_env/src/map_ros.cpp")
MAP_ROS_H = Path("src/planner/plan_env/include/plan_env/map_ros.h")


def test_over_depth_cache_extends_missing_clouds_for_eight_frames_only():
    text = MAP_ROS_CPP.read_text(encoding="utf-8")
    header = MAP_ROS_H.read_text(encoding="utf-8")

    assert "OVER_DEPTH_CACHE_MAX_MISSING_FRAMES = 8" in header
    assert "current_over_depth_empty" in text
    assert "cached_over_depth_cloud_" in text
    assert "agent.over_depth_missing_frames_ < OVER_DEPTH_CACHE_MAX_MISSING_FRAMES" in text
    assert "agent.cached_over_depth_cloud_.reset(new PointCloud3D(*agent.over_depth_object_cloud_));" in text
    assert "agent.over_depth_object_cloud_ = boost::make_shared<PointCloud3D>(*agent.cached_over_depth_cloud_);" in text
    assert "continue_over_depth_count_" not in header
