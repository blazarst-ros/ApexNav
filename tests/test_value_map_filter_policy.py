from pathlib import Path


VALUE_MAP_CPP = Path("src/planner/plan_env/src/value_map2d.cpp")


def test_value_map_uses_raw_itm_value_without_first_order_filter():
    text = VALUE_MAP_CPP.read_text(encoding="utf-8")

    assert "kNowValueGain" not in text
    assert "kPastValueGain" not in text
    assert "filtered_now_value" not in text
    assert "now_confidence * now_value + last_confidence * last_value" in text
