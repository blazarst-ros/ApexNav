from pathlib import Path


FSM_SOURCE = Path("src/planner/exploration_manager/src/exploration_fsm.cpp")
MANAGER_SOURCE = Path("src/planner/exploration_manager/src/exploration_manager.cpp")
MANAGER_HEADER = Path("src/planner/exploration_manager/include/exploration_manager/exploration_manager.h")
MAP_ROS_SOURCE = Path("src/planner/plan_env/src/map_ros.cpp")
MAP_ROS_HEADER = Path("src/planner/plan_env/include/plan_env/map_ros.h")
SDF_SOURCE = Path("src/planner/plan_env/src/sdf_map2d.cpp")
FSM_HEADER = Path("src/planner/exploration_manager/include/exploration_manager/exploration_fsm.h")
PYTHON_PARAMS = Path("params.py")


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
    helper_body = source[source.index("void ExplorationFSM::publishPlannerState()") :]

    assert "publishPlannerState();" in source
    assert helper_body.index("for (int agent_idx = 0; agent_idx < NUM_AGENTS; ++agent_idx)") < helper_body.index("ros_state_all_pub_.publish(state_all_msg);")


def test_python_detects_stale_planner_state_feedback():
    source = Path("habitat_evaluation.py").read_text(encoding="utf-8")

    assert "last_ros_state_update_time" in source
    assert "Planner state feedback is stale" in source


def test_episode_reset_publishes_state_ack_after_map_reset():
    source = FSM_SOURCE.read_text(encoding="utf-8")
    reset_body = source[source.index("void ExplorationFSM::resetEpisode()") :]

    assert "publishPlannerState();" in source
    assert reset_body.index("expl_manager_->resetEpisodeState();") < reset_body.index("publishPlannerState();")


def test_per_agent_exploration_results_are_published_as_array():
    source = FSM_SOURCE.read_text(encoding="utf-8")
    header = Path("src/planner/exploration_manager/include/exploration_manager/exploration_fsm.h").read_text(
        encoding="utf-8"
    )
    data_header = Path("src/planner/exploration_manager/include/exploration_manager/exploration_data.h").read_text(
        encoding="utf-8"
    )

    assert "expl_result_all_pub_" in header
    assert "expl_result_agent_pub_[NUM_AGENTS]" in header
    assert '"/ros/expl_result_all"' in source
    assert '"/ros/agent_" + std::to_string(i) + "/expl_result"' in source
    assert "void publishExplorationResults();" in header
    assert "void ExplorationFSM::publishExplorationResults()" in source
    assert "expl_result_all_msg.data.resize(NUM_AGENTS);" in source
    assert "expl_result_all_msg.data[agent_idx] = fd_->agent_[agent_idx].expl_result_;" in source
    assert "expl_result_agent_pub_[agent_idx].publish(expl_result_agent_msg);" in source
    assert "expl_result_all_pub_.publish(expl_result_all_msg);" in source
    assert "ad.expl_result_ = expl_res;" in source
    assert "expl_result_ = EXPL_RESULT::EXPLORATION;" in data_header


def test_python_reset_handshake_waits_longer_than_map_reset_before_retrying():
    source = Path("habitat_evaluation.py").read_text(encoding="utf-8")

    assert "RESET_ACK_TIMEOUT_SEC = 10.0" in source
    assert "RESET_STALE_TIMEOUT_SEC = 15.0" in source
    assert "wait_for_stale=RESET_STALE_TIMEOUT_SEC" in source


def test_failure_terminal_state_is_wire_compatible_between_cpp_and_python():
    header = FSM_HEADER.read_text(encoding="utf-8")
    params = PYTHON_PARAMS.read_text(encoding="utf-8")

    assert "FINISH = 5" in params
    assert "FINISH_FAILURE = 6" in params
    assert "FINISH = 5," in header
    assert "FINISH_FAILURE = 6" in header


def test_cpp_state_names_cover_failure_state_without_vector_indexing():
    source = FSM_SOURCE.read_text(encoding="utf-8")
    data_header = Path(
        "src/planner/exploration_manager/include/exploration_manager/exploration_data.h"
    ).read_text(encoding="utf-8")

    assert "stateName(pre_s)" in source
    assert "stateName(int(new_state))" in source
    assert "state_str_" not in data_header


def test_cpp_routes_planner_failures_to_local_failure_terminal_state():
    source = FSM_SOURCE.read_text(encoding="utf-8")

    assert "case ROS_STATE::FINISH_FAILURE:" in source
    assert 'transitState(agent_idx, ROS_STATE::FINISH_FAILURE, "Planner Failure")' in source
    assert 'transitState(stop_idx, ROS_STATE::FINISH, "Reach Object")' in source
    assert "broadcasting STOP to all agents" in source


def test_failure_terminal_state_is_idle_without_resetting_shared_maps():
    source = FSM_SOURCE.read_text(encoding="utf-8")
    failure_case = source[
        source.index("case ROS_STATE::FINISH_FAILURE:") :
        source.index("case ROS_STATE::PLAN_ACTION:")
    ]
    frontier_callback = source[source.index("void ExplorationFSM::frontierCallback") :]

    assert "action_pub_[agent_idx].publish(action_msg);" in failure_case
    assert "resetEpisode();" not in failure_case
    assert "ROS_STATE::FINISH_FAILURE" in frontier_callback


def test_python_reports_episode_before_requesting_planner_reset():
    source = Path("habitat_evaluation.py").read_text(encoding="utf-8")

    report_pos = source.index("print(table1)")
    record_pos = source.index("write_record(", report_pos)
    reset_pos = source.index("_finish_episode_handshake(", report_pos)

    assert report_pos < record_pos < reset_pos
