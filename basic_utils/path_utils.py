"""Small helpers for resolving repository-owned runtime assets."""

from pathlib import Path
from typing import Optional, Union


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent


def resolve_existing_path(
    path: Union[str, Path], *fallbacks: Optional[Union[str, Path]]
) -> str:
    """Resolve the first existing path, including project-relative fallbacks."""
    candidates = []
    seen = set()
    for raw_path in (path, *fallbacks):
        if raw_path in (None, ""):
            continue
        candidate = Path(raw_path).expanduser()
        raw_candidates = [candidate]
        if not candidate.is_absolute():
            raw_candidates = [Path.cwd() / candidate, PROJECT_ROOT / candidate, WORKSPACE_ROOT / candidate]
        for item in raw_candidates:
            resolved = item.resolve(strict=False)
            resolved_str = str(resolved)
            if resolved_str not in seen:
                seen.add(resolved_str)
                candidates.append(resolved)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    tried = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find required path. Tried:\n{tried}")


def scene_runtime_output_dir(base_dir: Union[str, Path], scene_id: str) -> Path:
    """Return the per-scene runtime directory using its three-character prefix."""
    scene_prefix = Path(scene_id).stem[:3]
    if not scene_prefix:
        raise ValueError("scene_id must contain a scene name")
    return Path(base_dir) / scene_prefix
