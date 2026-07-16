from pathlib import Path


HABITAT_EVAL = Path("habitat_evaluation.py")


def test_multiagent_success_requires_stop_semantics_not_distance_only():
    text = HABITAT_EVAL.read_text(encoding="utf-8")

    assert "def _is_successful_stop(" in text
    assert "return bool(stop_called) and distance_to_goal < success_distance" in text
    assert '"stop_called": False' in text
    assert 'ast["success"] = int(_is_successful_stop(' in text

    distance_block_start = text.index("if dtg <= success_distance:")
    distance_block_end = text.index("# Use task metrics", distance_block_start)
    distance_block = text[distance_block_start:distance_block_end]
    assert 'ast["success"] = 1' not in distance_block


def test_cooperative_policy_does_not_end_episode_on_single_agent_failure():
    text = HABITAT_EVAL.read_text(encoding="utf-8")

    cooperative_start = text.index('if termination_policy == "cooperative":')
    cooperative_end = text.index("else:", cooperative_start)
    cooperative_block = text[cooperative_start:cooperative_end]

    assert "mission_reached_object" in text
    assert "all_done = all(" in cooperative_block
    assert "if all_done:" in cooperative_block
    assert "break" in cooperative_block
    assert "any_finished = any(" not in cooperative_block
    assert "if any_finished:" not in cooperative_block
