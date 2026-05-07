from llm.utils.semantic_prior_defaults import ensure_target_semantic_prior


def test_ensure_target_semantic_prior_preserves_target_llm_value_only():
    priors = ensure_target_semantic_prior(
        "cabinet",
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
    assert set(priors.keys()) == {"cabinet"}


def test_ensure_target_semantic_prior_fills_missing_target():
    priors = ensure_target_semantic_prior("couch", {})

    assert priors["couch"]["mu_v"] == 0.85
    assert set(priors.keys()) == {"couch"}
