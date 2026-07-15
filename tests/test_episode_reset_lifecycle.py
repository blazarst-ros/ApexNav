from pathlib import Path


FSM_SOURCE = Path("src/planner/exploration_manager/src/exploration_fsm.cpp")
MANAGER_SOURCE = Path("src/planner/exploration_manager/src/exploration_manager.cpp")
MANAGER_HEADER = Path("src/planner/exploration_manager/include/exploration_manager/exploration_manager.h")
MAP_ROS_SOURCE = Path("src/planner/plan_env/src/map_ros.cpp")
MAP_ROS_HEADER = Path("src/planner/plan_env/include/plan_env/map_ros.h")
SDF_SOURCE = Path("src/planner/plan_env/src/sdf_map2d.cpp")


def test_episode_reset_clears_map_ros_agent_state_under_map_lock():
    header = MAP_ROS_HEADER.read_text(encoding="utf-8")
    source = MAP_ROS_SOURCE.read_text(encoding="utf-8")
    sdf_source = SDF_SOURCE.read_text(encoding="utf-8")

    assert "void resetEpisodeState();" in header
    assert "void MapROS::resetEpisodeState()" in source
    assert "std::lock_guard<std::mutex> lock(map_mutex_);" in source
    assert "agent.depth_cloud_->points.resize(640 * 480 / (skip_pixel_ * skip_pixel_));" in source
    assert "map_ros_->resetEpisodeState();" in sdf_source


def test_per_frame_virtual_ground_state_is_not_reused_across_agent_depth_updates():
    source = MAP_ROS_SOURCE.read_text(encoding="utf-8")

    assert "agent.filtered_depth_cloud2d_->clear();\n  agent.under_ground_cloud2d_->clear();" in source


def test_extreme_object_cache_is_resettable_not_function_static():
    source = MANAGER_SOURCE.read_text(encoding="utf-8")
    header = MANAGER_HEADER.read_text(encoding="utf-8")

    assert "static auto last_over_depth_object_cloud" not in source
    assert "last_over_depth_object_cloud_" in header
    assert "void resetEpisodeState();" in header
    assert "expl_manager_->resetEpisodeState();" in FSM_SOURCE.read_text(encoding="utf-8")


def test_visualization_never_indexes_object_labels_without_a_bound_check():
    source = FSM_SOURCE.read_text(encoding="utf-8")

    assert "const size_t object_count = std::min(ed_ptr->objects_.size(), ed_ptr->object_labels_.size());" in source
    assert "object label count mismatch" in source
    assert "for (size_t i = 0; i < object_count; ++i)" in source


def test_state_feedback_is_published_after_fsm_transitions():
    source = FSM_SOURCE.read_text(encoding="utf-8")

    assert source.index("for (int agent_idx = 0; agent_idx < NUM_AGENTS; ++agent_idx) {\n    state_all_msg.data[agent_idx] = state_[agent_idx];\n  }") < source.index("ros_state_all_pub_.publish(state_all_msg);")


def test_python_detects_stale_planner_state_feedback():
    source = Path("habitat_evaluation.py").read_text(encoding="utf-8")

    assert "last_ros_state_update_time" in source
    assert "Planner state feedback is stale" in source
