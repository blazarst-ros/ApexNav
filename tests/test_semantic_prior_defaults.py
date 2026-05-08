from llm.utils.semantic_prior_defaults import ensure_semantic_priors


def test_ensure_semantic_priors_preserves_llm_values_and_fills_missing():
    priors = ensure_semantic_priors(
        "cabinet",
        ["bookshelf", "dresser"],
        {
            "cabinet": {
                "category": "cabinet",
                "mu_v": 1.11,
                "sigma_v": 0.22,
                "unit": "meter",
                "rationale": "from llm",
            },
            "bookshelf": {
                "category": "bookshelf",
                "mu_v": 1.25,
                "sigma_v": 0.40,
                "unit": "meter",
                "rationale": "not target",
            }
        },
    )

    assert priors["cabinet"]["mu_v"] == 1.11
    assert priors["bookshelf"]["mu_v"] == 1.25
    assert priors["dresser"]["sigma_v"] == 0.35
    assert set(priors.keys()) == {"cabinet", "bookshelf", "dresser"}


def test_ensure_semantic_priors_fills_missing_target_and_confusions():
    priors = ensure_semantic_priors("couch", ["chair", "bed", "bench"], {})

    assert priors["couch"]["mu_v"] == 0.85
    assert priors["chair"]["mu_v"] == 0.85
    assert priors["bed"]["sigma_v"] == 0.30
    assert set(priors.keys()) == {"couch", "chair", "bed", "bench"}
