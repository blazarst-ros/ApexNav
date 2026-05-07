import json
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from llm.utils.semantic_prior_parser import (
    SemanticPriorMap,
    parse_semantic_verification_prior,
)
from llm.utils.semantic_prior_defaults import ensure_target_semantic_prior


STRUCTURED_FEEDBACK_PREFIX = "# LLM_FEEDBACK_JSON: "
SCHEMA_VERSION = 1


class LLMFeedbackRecord(TypedDict, total=False):
    schema_version: int
    label: str
    legacy_answer: List[Any]
    confusion_labels: List[str]
    fusion_score: float
    room: str
    semantic_verification_prior: SemanticPriorMap


def unpack_legacy_answer(legacy_answer: List[Any]) -> Tuple[List[str], str, float]:
    """Extract confusion labels, room, and score without mutating the legacy list."""
    items = list(legacy_answer or [])
    room = "unknown"
    fusion_score = 0.4

    if items:
        last_item = items[-1]
        if isinstance(last_item, str) and last_item not in [
            "move_forward",
            "turn_left",
            "turn_right",
            "stop",
        ]:
            room = str(items.pop())

    if items:
        last_item = items[-1]
        try:
            if isinstance(last_item, (float, int)):
                fusion_score = float(items.pop())
            elif isinstance(last_item, str) and last_item.replace(".", "", 1).isdigit():
                fusion_score = float(items.pop())
        except (TypeError, ValueError):
            fusion_score = 0.4

    confusion_labels = [str(item) for item in items if isinstance(item, str)]
    return confusion_labels, room, fusion_score


def build_feedback_record(
    label: str,
    legacy_answer: List[Any],
    raw_response: Optional[str] = None,
) -> LLMFeedbackRecord:
    confusion_labels, room, fusion_score = unpack_legacy_answer(legacy_answer)
    response = raw_response or ""
    semantic_priors = ensure_target_semantic_prior(
        label,
        parse_semantic_verification_prior(response),
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "label": label,
        "legacy_answer": list(legacy_answer or []),
        "confusion_labels": confusion_labels,
        "fusion_score": fusion_score,
        "room": room,
        "semantic_verification_prior": semantic_priors,
    }


def append_feedback_record(llm_answer_path: str, record: LLMFeedbackRecord) -> None:
    with open(llm_answer_path, "a+", encoding="utf-8") as f:
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True)
        f.write(f"\n{STRUCTURED_FEEDBACK_PREFIX}{payload}")


def read_feedback_records(
    llm_answer_path: str,
    label: Optional[str] = None,
) -> List[LLMFeedbackRecord]:
    records: List[LLMFeedbackRecord] = []
    try:
        with open(llm_answer_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return records

    for line in lines:
        if not line.startswith(STRUCTURED_FEEDBACK_PREFIX):
            continue

        payload = line[len(STRUCTURED_FEEDBACK_PREFIX) :].strip()
        try:
            record = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if not isinstance(record, dict):
            continue
        if label is not None and record.get("label") != label:
            continue
        records.append(record)
    return records


def read_latest_feedback_record(
    llm_answer_path: str,
    label: str,
) -> Optional[LLMFeedbackRecord]:
    records = read_feedback_records(llm_answer_path, label=label)
    return records[-1] if records else None
