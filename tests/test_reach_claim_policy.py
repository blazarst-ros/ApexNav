from basic_utils.failure_check.failure_check import get_reach_claim_outcome
from params import FINAL_RESULT


def test_reach_claim_is_false_positive_from_claiming_agents_current_distance():
    outcome = get_reach_claim_outcome(
        final_state=FINAL_RESULT.REACH_OBJECT,
        claim_agent_idx=1,
        claim_distance_to_goal=0.8779,
        success_distance=0.2,
        stop_executed=True,
    )

    assert outcome == "false positive"


def test_reach_claim_succeeds_only_after_stop_strictly_inside_habitat_range():
    assert (
        get_reach_claim_outcome(
            final_state=FINAL_RESULT.REACH_OBJECT,
            claim_agent_idx=1,
            claim_distance_to_goal=0.1999,
            success_distance=0.2,
            stop_executed=True,
        )
        == "success"
    )
    assert (
        get_reach_claim_outcome(
            final_state=FINAL_RESULT.REACH_OBJECT,
            claim_agent_idx=1,
            claim_distance_to_goal=0.2,
            success_distance=0.2,
            stop_executed=True,
        )
        == "false positive"
    )


def test_non_reach_final_state_has_no_reach_claim_outcome():
    assert (
        get_reach_claim_outcome(
            final_state=FINAL_RESULT.NO_FRONTIER,
            claim_agent_idx=1,
            claim_distance_to_goal=0.1,
            success_distance=0.2,
            stop_executed=True,
        )
        is None
    )
