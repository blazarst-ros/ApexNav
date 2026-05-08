import json

from llm.answer_reader.structured_feedback import (
    STRUCTURED_FEEDBACK_PREFIX,
    append_feedback_record,
    build_feedback_record,
    read_feedback_records,
    read_latest_feedback_record,
    unpack_legacy_answer,
)


def test_unpack_legacy_answer_without_mutation():
    legacy_answer = ["bookshelf", "dresser", "closet", 0.50, "kitchen"]

    confusion_labels, room, fusion_score = unpack_legacy_answer(legacy_answer)

    assert legacy_answer == ["bookshelf", "dresser", "closet", 0.50, "kitchen"]
    assert confusion_labels == ["bookshelf", "dresser", "closet"]
    assert room == "kitchen"
    assert fusion_score == 0.50


def test_build_feedback_record_includes_semantic_prior():
    response = """
Answer: [bookshelf, dresser, closet, 0.50, kitchen]

Semantic Verification Prior:
[
  {category: cabinet, mu_v: 1.05, sigma_v: 0.35, unit: meter, rationale: mid-height storage},
  {category: bookshelf, mu_v: 1.25, sigma_v: 0.40, unit: meter, rationale: tall vertical storage}
]
"""

    record = build_feedback_record(
        "cabinet",
        ["bookshelf", "dresser", "closet", 0.50, "kitchen"],
        response,
    )

    assert record["label"] == "cabinet"
    assert record["confusion_labels"] == ["bookshelf", "dresser", "closet"]
    assert record["fusion_score"] == 0.50
    assert record["room"] == "kitchen"
    assert record["semantic_verification_prior"]["cabinet"]["mu_v"] == 1.05
    assert record["semantic_verification_prior"]["bookshelf"]["sigma_v"] == 0.40
    assert record["semantic_verification_prior"]["dresser"]["mu_v"] == 1.05
    assert record["semantic_verification_prior"]["closet"]["sigma_v"] == 0.45
    assert set(record["semantic_verification_prior"].keys()) == {
        "cabinet",
        "bookshelf",
        "dresser",
        "closet",
    }
    assert "created_at_utc" not in record
    assert "raw_response" not in record


def test_build_feedback_record_adds_default_priors_when_llm_prior_missing():
    record = build_feedback_record(
        "couch",
        ["chair", "bed", "bench", 0.45, "living room"],
        "",
    )

    priors = record["semantic_verification_prior"]
    assert priors["couch"]["mu_v"] == 0.85
    assert priors["couch"]["sigma_v"] == 0.30
    assert priors["chair"]["mu_v"] == 0.85
    assert priors["bed"]["sigma_v"] == 0.30
    assert priors["bench"]["unit"] == "meter"
    assert set(priors.keys()) == {"couch", "chair", "bed", "bench"}


def test_append_and_read_feedback_records(tmp_path):
    cache_path = tmp_path / "llm_answer_hm3d.txt"
    cache_path.write_text("cabinet: ['bookshelf', 0.5, 'kitchen']\n", encoding="utf-8")

    record = build_feedback_record(
        "cabinet",
        ["bookshelf", 0.50, "kitchen"],
        "Semantic Verification Prior: [{category: cabinet, mu_v: 1.0, sigma_v: 0.3}]",
    )
    append_feedback_record(str(cache_path), record)

    lines = cache_path.read_text(encoding="utf-8").splitlines()
    structured_lines = [line for line in lines if line.startswith(STRUCTURED_FEEDBACK_PREFIX)]
    payload = json.loads(structured_lines[0][len(STRUCTURED_FEEDBACK_PREFIX) :])

    assert payload["label"] == "cabinet"
    assert read_feedback_records(str(cache_path), label="cabinet")[0]["label"] == "cabinet"
    assert read_latest_feedback_record(str(cache_path), "cabinet")["room"] == "kitchen"
