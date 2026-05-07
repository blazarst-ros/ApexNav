import ast
import json
import re
from typing import Dict, List, Optional, TypedDict


class SemanticVerificationPrior(TypedDict, total=False):
    category: str
    mu_v: float
    sigma_v: float
    unit: str
    rationale: str


SemanticPriorMap = Dict[str, SemanticVerificationPrior]


_SECTION_RE = re.compile(
    r"Semantic\s+Verification\s+Prior\s*:\s*(\[[\s\S]*?\])",
    re.IGNORECASE,
)
_OBJECT_RE = re.compile(r"\{([^{}]*)\}", re.DOTALL)
_PAIR_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)(?=,\s*[A-Za-z_][A-Za-z0-9_]*\s*:|$)",
    re.DOTALL,
)


def parse_semantic_verification_prior(response: Optional[str]) -> SemanticPriorMap:
    """Parse the optional semantic prior block without affecting legacy Answer parsing."""
    if not response or not isinstance(response, str):
        return {}

    block = _extract_prior_block(response)
    if not block:
        return {}

    parsed_items = _parse_json_like_list(block)
    if parsed_items is None:
        parsed_items = _parse_loose_objects(block)

    priors: SemanticPriorMap = {}
    for item in parsed_items:
        prior = _normalize_prior(item)
        if prior is not None:
            priors[prior["category"]] = prior
    return priors


def _extract_prior_block(response: str) -> str:
    match = _SECTION_RE.search(response)
    return match.group(1).strip() if match else ""


def _parse_json_like_list(block: str) -> Optional[List[dict]]:
    for candidate in (block, _quote_unquoted_keys(block)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(candidate)
            except (SyntaxError, ValueError):
                continue

        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    return None


def _quote_unquoted_keys(text: str) -> str:
    return re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', text)


def _parse_loose_objects(block: str) -> List[dict]:
    objects: List[dict] = []
    for object_match in _OBJECT_RE.finditer(block):
        item = {}
        for key, value in _PAIR_RE.findall(object_match.group(1)):
            item[key.strip()] = _clean_value(value)
        if item:
            objects.append(item)
    return objects


def _clean_value(value: object) -> object:
    if not isinstance(value, str):
        return value

    cleaned = value.strip().strip(",").strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ("'", '"'):
        cleaned = cleaned[1:-1]

    try:
        return float(cleaned)
    except ValueError:
        return cleaned


def _normalize_prior(item: dict) -> Optional[SemanticVerificationPrior]:
    category = str(item.get("category", "")).strip()
    if not category:
        return None

    try:
        mu_v = float(item["mu_v"])
        sigma_v = float(item["sigma_v"])
    except (KeyError, TypeError, ValueError):
        return None

    if mu_v <= 0.0 or sigma_v <= 0.0:
        return None

    return {
        "category": category,
        "mu_v": mu_v,
        "sigma_v": sigma_v,
        "unit": str(item.get("unit", "meter")).strip() or "meter",
        "rationale": str(item.get("rationale", "")).strip(),
    }
