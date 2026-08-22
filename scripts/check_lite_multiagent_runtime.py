#!/usr/bin/env python3
"""Fail fast when the Lite Habitat runtime cannot host MultiAgentSim-v0."""

import importlib
import importlib.metadata
import subprocess
import sys
from pathlib import Path


EXPECTED_HABITAT_SIM_VERSION = "0.3.1"
DEFAULT_PATCH_FILE = (
    Path(__file__).resolve().parents[1]
    / "patches/habitat-lab-v0.3.1-multi-agent.patch"
)


def _habitat_sim_version() -> str:
    habitat_sim = importlib.import_module("habitat_sim")
    version = getattr(habitat_sim, "__version__", None)
    if version is None:
        version = importlib.metadata.version("habitat-sim")
    return str(version)


def _check_complete_patch(local_habitat: Path, patch_file: Path) -> None:
    patch_file = patch_file.resolve()
    if not patch_file.is_file():
        raise RuntimeError(f"Habitat multi-agent patch file is missing: {patch_file}")

    habitat_checkout = local_habitat.resolve().parent
    result = subprocess.run(
        [
            "git",
            "-C",
            str(habitat_checkout),
            "apply",
            "--unidiff-zero",
            "--reverse",
            "--check",
            str(patch_file),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "reverse-check failed"
        raise RuntimeError(
            "complete Habitat multi-agent patch is not applied to "
            f"{habitat_checkout}: {detail}"
        )


def check_runtime(
    local_habitat: Path,
    patch_file: Path = DEFAULT_PATCH_FILE,
) -> None:
    version = _habitat_sim_version()
    if version != EXPECTED_HABITAT_SIM_VERSION:
        raise RuntimeError(
            f"habitat-sim {version} is incompatible; "
            f"Lite ApexNav requires {EXPECTED_HABITAT_SIM_VERSION}"
        )

    expected_module = (
        local_habitat
        / "habitat/sims/habitat_simulator/multi_agent_sim.py"
    ).resolve()
    if not expected_module.is_file():
        raise RuntimeError(f"local multi-agent simulator is missing: {expected_module}")

    _check_complete_patch(local_habitat, patch_file)

    module = importlib.import_module(
        "habitat.sims.habitat_simulator.multi_agent_sim"
    )
    loaded_module = Path(module.__file__).resolve()
    if loaded_module != expected_module:
        raise RuntimeError(
            "MultiAgentSim was imported from the wrong Habitat checkout: "
            f"{loaded_module} (expected {expected_module})"
        )

    registry = importlib.import_module("habitat.core.registry").registry
    if registry.get_simulator("MultiAgentSim-v0") is None:
        raise RuntimeError("MultiAgentSim-v0 is not registered by the local Habitat patch")


def main(argv) -> int:
    if len(argv) != 3:
        print(
            f"Usage: {argv[0]} <local-habitat-lab-package> <multi-agent-patch>",
            file=sys.stderr,
        )
        return 2
    try:
        check_runtime(Path(argv[1]).resolve(), Path(argv[2]).resolve())
    except Exception as exc:
        print(f"Lite Habitat preflight failed: {exc}", file=sys.stderr)
        return 1
    print(
        "Lite Habitat preflight OK: "
        f"habitat-sim {EXPECTED_HABITAT_SIM_VERSION}, MultiAgentSim-v0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
