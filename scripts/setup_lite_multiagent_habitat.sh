#!/usr/bin/env bash
# Install the reproducible MultiAgentSim-v0 patch into Habitat-Lab v0.3.1.
set -euo pipefail

readonly HABITAT_BASE="142616776544f918c19e7f0392b65cc8cc69fa13"
readonly APEXNAV_ROOT="${APEXNAV_ROOT:-/home/blazarst/ApexNav}"
readonly HABITAT_LAB_DIR="${HABITAT_LAB_DIR:-$APEXNAV_ROOT/habitat-lab}"
readonly PATCH_FILE="${PATCH_FILE:-$APEXNAV_ROOT/patches/habitat-lab-v0.3.1-multi-agent.patch}"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

[[ -d "$HABITAT_LAB_DIR/.git" || -f "$HABITAT_LAB_DIR/.git" ]] || \
    fail "Habitat-Lab checkout is missing: $HABITAT_LAB_DIR. Clone v0.3.1 there, then rerun this script."
[[ -f "$PATCH_FILE" ]] || \
    fail "Multi-agent patch is missing: $PATCH_FILE. Restore the tracked patches/habitat-lab-v0.3.1-multi-agent.patch file."

actual_base="$(git -C "$HABITAT_LAB_DIR" rev-parse HEAD 2>/dev/null)" || \
    fail "Cannot read the Habitat-Lab HEAD in $HABITAT_LAB_DIR."
if [[ "$actual_base" != "$HABITAT_BASE" ]]; then
    fail "Habitat-Lab must be at base commit $HABITAT_BASE, but HEAD is $actual_base. Recover with: git -C $HABITAT_LAB_DIR checkout --detach $HABITAT_BASE"
fi

if git -C "$HABITAT_LAB_DIR" apply --reverse --check "$PATCH_FILE"; then
    printf 'MultiAgentSim-v0 patch is already applied to %s.\n' "$HABITAT_LAB_DIR"
    exit 0
fi

if ! git -C "$HABITAT_LAB_DIR" diff --quiet || [[ -n "$(git -C "$HABITAT_LAB_DIR" ls-files --others --exclude-standard)" ]]; then
    fail "Habitat-Lab has local changes that are not this patch. Preserve them, or use a clean v0.3.1 checkout before applying: git -C $HABITAT_LAB_DIR status --short"
fi

if ! git -C "$HABITAT_LAB_DIR" apply --check "$PATCH_FILE"; then
    fail "The patch cannot be applied cleanly. Confirm this is the exact v0.3.1 base and inspect with: git -C $HABITAT_LAB_DIR apply --check $PATCH_FILE"
fi

git -C "$HABITAT_LAB_DIR" apply "$PATCH_FILE"
printf 'Applied MultiAgentSim-v0 patch to %s.\n' "$HABITAT_LAB_DIR"
