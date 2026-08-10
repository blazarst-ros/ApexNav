from pathlib import Path


def test_runtime_topology_is_limited_to_two_agents():
    two_agent_files = [
        Path("CLAUDE.md"),
        Path("README.md"),
        Path("Op.txt"),
        Path("RuntimeData/README.md"),
        Path("RuntimeData/capture_ros_data.sh"),
        Path("habitat2ros/habitat_publisher.py"),
        Path("real_world_test_example/config/real_world_test.yaml"),
        Path("src/planner/exploration_manager/config/ApexNav.rviz"),
        Path("src/planner/exploration_manager/config/ApexNav_Traj.rviz"),
        Path("src/planner/exploration_manager/launch/algorithm.xml"),
        Path("src/planner/plan_env/include/plan_env/map_ros.h"),
    ]

    forbidden_terms = (
        "agent_2",
        "agent2_",
        "NUM_AGENTS_ = 3",
        "NUM_AGENTS = 3",
        "三机",
        "3 agents",
    )
    for path in two_agent_files:
        text = path.read_text(encoding="utf-8")
        assert not any(term in text for term in forbidden_terms), path

    assert "agents=(0 1)" in Path("RuntimeData/capture_ros_data.sh").read_text(encoding="utf-8")

    for path in [
        Path("config/habitat_eval_hm3dv1.yaml"),
        Path("config/habitat_eval_hm3dv2.yaml"),
        Path("config/habitat_eval_mp3d.yaml"),
    ]:
        text = path.read_text(encoding="utf-8")
        assert "num_agents: 2" in text, path
        assert "perception_agents_per_step: 2" in text, path
        assert "agent_2:" not in text, path


def test_real_world_example_configures_two_agents():
    text = Path("real_world_test_example/config/real_world_test.yaml").read_text(encoding="utf-8")
    assert "num_agents: 2" in text
    assert "agent_0:" in text
    assert "agent_1:" in text
    assert "agent_2:" not in text


def test_map_ros_exposes_the_pitch_angle_used_by_object_filtering():
    header = Path("src/planner/plan_env/include/plan_env/map_ros.h").read_text(encoding="utf-8")
    source = Path("src/planner/plan_env/src/map_ros.cpp").read_text(encoding="utf-8")
    capture = Path("RuntimeData/capture_ros_data.sh").read_text(encoding="utf-8")

    assert "camera_pitch_pub_" in header
    assert '"/map_ros/agent_" + std::to_string(id) + "/camera_pitch"' in source
    assert "camera_pitch_pub_[agent_id].publish(camera_pitch_msg);" in source
    assert "pitch)" in capture
    assert 'topics+=("/map_ros/agent_${agent}/camera_pitch")' in capture


def test_multi_agent_fsm_waits_for_action_finish_without_republishing_actions():
    source = Path("src/planner/exploration_manager/src/exploration_fsm.cpp").read_text(
        encoding="utf-8"
    )
    data_header = Path(
        "src/planner/exploration_manager/include/exploration_manager/exploration_data.h"
    ).read_text(encoding="utf-8")
    fsm_header = Path(
        "src/planner/exploration_manager/include/exploration_manager/exploration_fsm.h"
    ).read_text(encoding="utf-8")

    wait_block = source.split("case ROS_STATE::WAIT_ACTION_FINISH: {")[1].split(
        "case ROS_STATE::", 1
    )[0]
    assert "action_pub_[agent_idx].publish" not in wait_block
    assert "wait_action_finish_count_" not in source
    assert "wait_action_finish_count_" not in data_header
    assert "MAX_WAIT_ACTION_FINISH" not in fsm_header
