from llm.utils.answer_validation import (
    get_fallback_answer,
    is_valid_legacy_answer,
    split_legacy_answer,
)


def test_stop_cache_entry_is_invalid():
    assert not is_valid_legacy_answer(["stop"])


def test_valid_answer_splits_without_mutation():
    answer = ["chair", "bed", "bench", 0.45, "living room"]

    labels, room, score, score_found = split_legacy_answer(answer)

    assert answer == ["chair", "bed", "bench", 0.45, "living room"]
    assert labels == ["chair", "bed", "bench"]
    assert room == "living room"
    assert score == 0.45
    assert score_found


def test_couch_fallback_is_valid():
    fallback = get_fallback_answer("couch")

    assert fallback == ["chair", "bed", "bench", 0.45, "living room"]
    assert is_valid_legacy_answer(fallback)
