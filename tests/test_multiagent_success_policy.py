from pathlib import Path

import basic_utils.failure_check.failure_check as failure_policy

from params import ROS_STATE


HABITAT_EVAL = Path("habitat_evaluation.py")


def test_multiagent_success_requires_stop_semantics_not_distance_only():
    text = HABITAT_EVAL.read_text(encoding="utf-8")

    assert "def _is_successful_stop(" in text
    assert "return bool(stop_called) and distance_to_goal < success_distance" in text
    assert '"stop_called": False' in text
    assert 'ast["success"] = int(_is_successful_stop(' in text
    assert 'rospy.Subscriber("/ros/reach_claim", Int32MultiArray, ros_reach_claim_callback' in text
    assert 'reach_claim_distance = _get_agent_distance_to_goal(' in text
    assert 'ast["distance_to_goal"] = dtg' in text
    assert 'min(ast["distance_to_goal"], dtg)' not in text
    assert '"Reach Claim Agent"' in text
    assert '"Reach Claim Distance"' in text
    assert '"Reach Claim Outcome"' in text

    distance_block_start = text.index("if dtg < success_distance:")
    distance_block_end = text.index("# Use task metrics", distance_block_start)
    distance_block = text[distance_block_start:distance_block_end]
    assert 'ast["success"] = 1' not in distance_block


def test_cooperative_policy_does_not_end_episode_on_single_agent_failure():
    should_end = failure_policy.should_end_multiagent_episode(
        ros_states=[ROS_STATE.FINISH_FAILURE, ROS_STATE.PLAN_ACTION],
        step_counts=[73, 73],
        max_episode_steps=250,
        reach_stop_queued=False,
        reach_claim_stop_executed=False,
    )

    assert should_end is False


def test_cooperative_policy_ends_when_every_agent_failed():
    should_end = failure_policy.should_end_multiagent_episode(
        ros_states=[ROS_STATE.FINISH_FAILURE, ROS_STATE.FINISH_FAILURE],
        step_counts=[73, 121],
        max_episode_steps=250,
        reach_stop_queued=False,
        reach_claim_stop_executed=False,
    )

    assert should_end is True


def test_reach_claim_waits_for_habitat_stop_execution():
    waiting = failure_policy.should_end_multiagent_episode(
        ros_states=[ROS_STATE.FINISH, ROS_STATE.FINISH],
        step_counts=[73, 121],
        max_episode_steps=250,
        reach_stop_queued=True,
        reach_claim_stop_executed=False,
    )
    stopped = failure_policy.should_end_multiagent_episode(
        ros_states=[ROS_STATE.FINISH, ROS_STATE.FINISH],
        step_counts=[73, 121],
        max_episode_steps=250,
        reach_stop_queued=True,
        reach_claim_stop_executed=True,
    )

    assert waiting is False
    assert stopped is True


def test_all_agents_at_habitat_step_limit_remains_a_hard_termination():
    should_end = failure_policy.should_end_multiagent_episode(
        ros_states=[ROS_STATE.PLAN_ACTION, ROS_STATE.WAIT_ACTION_FINISH],
        step_counts=[250, 250],
        max_episode_steps=250,
        reach_stop_queued=False,
        reach_claim_stop_executed=False,
    )

    assert should_end is True


def test_multiagent_termination_reason_distinguishes_team_exit_causes():
    reason = failure_policy.get_multiagent_termination_reason

    assert reason(
        [ROS_STATE.FINISH_FAILURE, ROS_STATE.PLAN_ACTION],
        [73, 74], 250, False, False,
    ) is None
    assert reason(
        [ROS_STATE.FINISH_FAILURE, ROS_STATE.FINISH_FAILURE],
        [73, 121], 250, False, False,
    ) == "all_failed"
    assert reason(
        [ROS_STATE.FINISH_FAILURE, ROS_STATE.WAIT_ACTION_FINISH],
        [73, 250], 250, False, False,
    ) == "step_limit"
    assert reason(
        [ROS_STATE.FINISH, ROS_STATE.FINISH],
        [73, 121], 250, True, True,
    ) == "reach_claim"
    assert reason(
        [ROS_STATE.FINISH, ROS_STATE.FINISH],
        [73, 121], 250, True, False,
    ) is None


def test_multiagent_uses_per_agent_final_results_for_reporting():
    text = HABITAT_EVAL.read_text(encoding="utf-8")

    assert 'rospy.Subscriber("/ros/final_result_all", Int32MultiArray' in text
    assert "ros_final_results = [FINAL_RESULT.EXPLORE] * num_agents" in text
    assert "report_final_state = ros_final_results[report_agent_idx]" in text
    assert "report_final_state," in text


def test_multiagent_stop_is_not_sent_to_simulator_move_fn():
    text = HABITAT_EVAL.read_text(encoding="utf-8")

    assert "active_sim_actions = {" in text
    sim_action_block_start = text.index("active_sim_actions = {")
    sim_action_block_end = text.index("if active_sim_actions:", sim_action_block_start)
    sim_action_block = text[sim_action_block_start:sim_action_block_end]

    assert "v != HabitatSimActions.stop" in sim_action_block
    assert "_multi_agent_step(env, active_sim_actions, agent_names)" in text
    assert "measure_action = HabitatSimActions.stop" in text
