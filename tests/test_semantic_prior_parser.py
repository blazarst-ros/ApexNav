from llm.utils.only_answer import only_answer
from llm.utils.semantic_prior_parser import parse_semantic_verification_prior


def test_legacy_answer_parser_ignores_semantic_prior_block():
    response = """
Answer: [donut, pizza, sandwich, pie, 0.30, everywhere]

Semantic Verification Prior:
[
  {category: cake, mu_v: 0.95, sigma_v: 0.25, unit: meter, rationale: tabletop object}
]
"""

    assert only_answer(response) == ["donut", "pizza", "sandwich", "pie", 0.30, "everywhere"]


def test_parse_loose_semantic_prior_block():
    response = """
Answer: [bookshelf, dresser, closet, 0.50, kitchen]

Semantic Verification Prior:
[
  {category: dining table, mu_v: 1.10, sigma_v: 0.30, unit: meter, rationale: elevated tabletop verification},
  {category: bookshelf, mu_v: 1.25, sigma_v: 0.40, unit: meter, rationale: tall vertical storage}
]
"""

    priors = parse_semantic_verification_prior(response)

    assert priors["dining table"]["mu_v"] == 1.10
    assert priors["dining table"]["sigma_v"] == 0.30
    assert priors["bookshelf"]["unit"] == "meter"
    assert "vertical" in priors["bookshelf"]["rationale"]


def test_parse_valid_json_semantic_prior_block():
    response = """
Semantic Verification Prior:
[
  {"category": "chair", "mu_v": 0.85, "sigma_v": 0.20, "unit": "meter", "rationale": "seat and backrest"}
]
"""

    priors = parse_semantic_verification_prior(response)

    assert priors == {
        "chair": {
            "category": "chair",
            "mu_v": 0.85,
            "sigma_v": 0.20,
            "unit": "meter",
            "rationale": "seat and backrest",
        }
    }


def test_missing_or_invalid_prior_returns_empty_map():
    assert parse_semantic_verification_prior("Answer: [chair, 0.4, living room]") == {}
    assert parse_semantic_verification_prior(None) == {}
    assert parse_semantic_verification_prior("Semantic Verification Prior: [{category: chair}]") == {}
