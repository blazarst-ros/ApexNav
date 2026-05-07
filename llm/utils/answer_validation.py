from typing import Any, Dict, List, Tuple


DEFAULT_FUSION_SCORE = 0.4
DEFAULT_ROOM = "unknown"
ACTION_TOKENS = {"move_forward", "turn_left", "turn_right", "stop"}


FALLBACK_ANSWERS: Dict[str, List[Any]] = {
    "couch": ["chair", "bed", "bench", 0.45, "living room"],
    "sofa": ["chair", "bed", "bench", 0.45, "living room"],
    "cabinet": ["bookshelf", "dresser", "closet", 0.50, "kitchen"],
    "bed": ["couch", "bench", "dining table", 0.60, "bedroom"],
    "chair": ["couch", "bench", "dining table", 0.40, "everywhere"],
    "dining table": ["desk", "counter", "bench", 0.55, "dining room"],
    "potted plant": ["vase", "lamp", "chair", 0.35, "everywhere"],
    "counter": ["dining table", "desk", "shelf", 0.45, "kitchen"],
}


def split_legacy_answer(answer: List[Any]) -> Tuple[List[Any], str, float, bool]:
    items = list(answer or [])
    room = DEFAULT_ROOM
    fusion_score = DEFAULT_FUSION_SCORE

    if items:
        last_item = items[-1]
        if isinstance(last_item, str) and last_item not in ACTION_TOKENS:
            room = str(items.pop())

    score_found = False
    if items:
        last_item = items[-1]
        try:
            if isinstance(last_item, (float, int)):
                fusion_score = float(items.pop())
                score_found = True
            elif isinstance(last_item, str) and last_item.replace(".", "", 1).isdigit():
                fusion_score = float(items.pop())
                score_found = True
        except (TypeError, ValueError):
            score_found = False

    return items, room, fusion_score, score_found


def is_valid_legacy_answer(answer: Any) -> bool:
    if not isinstance(answer, list) or len(answer) < 3:
        return False

    confusion_labels, room, fusion_score, score_found = split_legacy_answer(answer)
    if not score_found:
        return False
    if room == DEFAULT_ROOM:
        return False
    if not 0.0 <= fusion_score <= 1.0:
        return False

    valid_labels = [
        item
        for item in confusion_labels
        if isinstance(item, str) and item and item not in ACTION_TOKENS
    ]
    return len(valid_labels) >= 1


def get_fallback_answer(label: str) -> List[Any]:
    normalized = (label or "").strip().lower()
    if normalized in FALLBACK_ANSWERS:
        return list(FALLBACK_ANSWERS[normalized])
    return ["chair", "cabinet", "table", 0.40, "everywhere"]
