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
