import ast
import gzip
import json
from pathlib import Path

from llm.answer_reader.structured_feedback import STRUCTURED_FEEDBACK_PREFIX
from vlm.Labels import MP3D_ID_TO_NAME


ANSWER_FILES = [
    Path("llm/answers/llm_answer_hm3d.txt"),
    Path("llm/answers/llm_answer_mp3d.txt"),
]


def _runtime_labels():
    with gzip.open(
        "data/datasets/objectnav/mp3d/v1/val/val.json.gz",
        "rt",
        encoding="utf-8",
    ) as file:
        data = json.load(file)
    return [MP3D_ID_TO_NAME[idx] for idx, _ in enumerate(data["category_to_mp3d_category_id"])]


def _parse_answer_file(path):
    legacy = {}
    structured = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            if line.startswith(STRUCTURED_FEEDBACK_PREFIX):
                payload = json.loads(line[len(STRUCTURED_FEEDBACK_PREFIX) :])
                structured[payload["label"]] = payload
            continue
        label, raw_answer = line.split(": ", 1)
        legacy[label] = ast.literal_eval(raw_answer)
    return legacy, structured


def test_llm_answer_files_cover_all_runtime_labels_with_structured_priors():
    labels = _runtime_labels()
    for answer_file in ANSWER_FILES:
        legacy, structured = _parse_answer_file(answer_file)
        assert sorted(legacy) == sorted(labels)
        assert sorted(structured) == sorted(labels)
        for label in labels:
            answer = legacy[label]
            record = structured[label]
            assert answer == record["legacy_answer"]
            assert record["label"] == label
            assert 0.25 <= record["fusion_score"] <= 0.65
            assert record["room"] != "unknown"
            categories = set(record["semantic_verification_prior"])
            assert label in categories
            assert set(record["confusion_labels"]).issubset(categories)
