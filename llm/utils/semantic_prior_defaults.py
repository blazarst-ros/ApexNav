from llm.utils.semantic_prior_parser import SemanticPriorMap, SemanticVerificationPrior


DEFAULT_PRIOR = {
    "mu_v": 1.00,
    "sigma_v": 0.35,
    "unit": "meter",
    "rationale": "Default fallback prior for indoor object verification.",
}


CATEGORY_PRIOR_DEFAULTS = {
    "bed": {
        "mu_v": 0.90,
        "sigma_v": 0.30,
        "rationale": "Beds are low large furniture, best verified from low-to-mid camera height.",
    },
    "bench": {
        "mu_v": 0.85,
        "sigma_v": 0.30,
        "rationale": "Benches are seat-height furniture, best verified around seat/backrest height.",
    },
    "bookshelf": {
        "mu_v": 1.25,
        "sigma_v": 0.40,
        "rationale": "Bookshelves are tall vertical storage furniture.",
    },
    "cabinet": {
        "mu_v": 1.05,
        "sigma_v": 0.35,
        "rationale": "Cabinets are mid-height storage furniture with doors, shelves, and top edges.",
    },
    "chair": {
        "mu_v": 0.85,
        "sigma_v": 0.30,
        "rationale": "Chairs are seat-level objects with backrest and legs.",
    },
    "closet": {
        "mu_v": 1.35,
        "sigma_v": 0.45,
        "rationale": "Closets are tall storage regions and benefit from a higher viewpoint.",
    },
    "couch": {
        "mu_v": 0.85,
        "sigma_v": 0.30,
        "rationale": "Couches are low seat-level furniture, best verified from low-to-mid camera height.",
    },
    "counter": {
        "mu_v": 1.05,
        "sigma_v": 0.30,
        "rationale": "Counters are horizontal surfaces usually near waist height.",
    },
    "dining table": {
        "mu_v": 1.10,
        "sigma_v": 0.30,
        "rationale": "Dining tables are large horizontal surfaces with legs and tabletop extent.",
    },
    "dresser": {
        "mu_v": 1.05,
        "sigma_v": 0.35,
        "rationale": "Dressers are mid-height storage furniture with drawer structure.",
    },
    "potted plant": {
        "mu_v": 0.95,
        "sigma_v": 0.35,
        "rationale": "Potted plants vary in size but are commonly verified around pot and foliage height.",
    },
    "refrigerator": {
        "mu_v": 1.25,
        "sigma_v": 0.40,
        "rationale": "Refrigerators are tall box-like objects requiring vertical extent verification.",
    },
    "sofa": {
        "mu_v": 0.85,
        "sigma_v": 0.30,
        "rationale": "Sofas are low seat-level furniture, similar to couches.",
    },
}


def get_default_semantic_prior(category: str) -> SemanticVerificationPrior:
    normalized = (category or "").strip().lower()
    defaults = dict(DEFAULT_PRIOR)
    defaults.update(CATEGORY_PRIOR_DEFAULTS.get(normalized, {}))
    return {
        "category": category,
        "mu_v": float(defaults["mu_v"]),
        "sigma_v": float(defaults["sigma_v"]),
        "unit": str(defaults["unit"]),
        "rationale": str(defaults["rationale"]),
    }


def ensure_target_semantic_prior(
    target_label: str,
    parsed_priors: SemanticPriorMap,
) -> SemanticPriorMap:
    priors = dict(parsed_priors or {})
    if target_label in priors:
        return {target_label: priors[target_label]}
    return {target_label: get_default_semantic_prior(target_label)}
