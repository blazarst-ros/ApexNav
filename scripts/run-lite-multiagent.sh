#!/usr/bin/env bash
# Execute a Lite-ApexNav command with the local patched Habitat-Lab first.
set -euo pipefail

readonly APEXNAV_ROOT="${APEXNAV_ROOT:-/home/blazarst/ApexNav}"
readonly LITE_APEX_ROOT="${LITE_APEX_ROOT:-/media/blazarst/Getea/Lite-ApexNav}"
readonly LITE_PYTHON="${LITE_PYTHON:-$LITE_APEX_ROOT/conda-env/Lite-apex/bin/python}"
readonly ROS_SETUP="${ROS_SETUP:-/opt/ros/noetic/setup.bash}"
readonly DEVEL_SETUP="${DEVEL_SETUP:-$APEXNAV_ROOT/devel/setup.bash}"
readonly LOCAL_HABITAT="$APEXNAV_ROOT/habitat-lab/habitat-lab"
readonly LITE_PREFLIGHT="${LITE_PREFLIGHT:-$APEXNAV_ROOT/scripts/check_lite_multiagent_runtime.py}"
readonly PATCH_FILE="${PATCH_FILE:-$APEXNAV_ROOT/patches/habitat-lab-v0.3.1-multi-agent.patch}"

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

[[ "$#" -gt 0 ]] || fail "Usage: $0 <python-script-or-module-argument> [arguments ...]"
[[ -x "$LITE_PYTHON" ]] || fail "Lite Python is missing: $LITE_PYTHON. Check the Lite-ApexNav environment."
[[ -d "$LOCAL_HABITAT" ]] || fail "Local Habitat-Lab source is missing: $LOCAL_HABITAT. Run scripts/setup_lite_multiagent_habitat.sh first."
[[ -f "$LITE_PREFLIGHT" ]] || fail "Lite runtime preflight is missing: $LITE_PREFLIGHT. Restore the ApexNav scripts directory."
[[ -f "$PATCH_FILE" ]] || fail "Habitat multi-agent patch is missing: $PATCH_FILE. Restore the tracked patch and run scripts/setup_lite_multiagent_habitat.sh."
[[ -f "$ROS_SETUP" ]] || fail "ROS setup script is missing: $ROS_SETUP. Install/source ROS Noetic before running."
[[ -f "$DEVEL_SETUP" ]] || fail "ApexNav devel setup is missing: $DEVEL_SETUP. Build the ROS workspace with catkin_make first."

source "$ROS_SETUP"
source "$DEVEL_SETUP"

export LD_PRELOAD="$LITE_APEX_ROOT/conda-env/Lite-apex/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH="$LOCAL_HABITAT:$APEXNAV_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export YOLOE_WEIGHTS="${YOLOE_WEIGHTS:-$LITE_APEX_ROOT/model-cache/yoloe-11l-seg.pt}"
export CLIP_DOWNLOAD_ROOT="${CLIP_DOWNLOAD_ROOT:-$LITE_APEX_ROOT/model-cache/clip}"
export TMPDIR="${TMPDIR:-$LITE_APEX_ROOT/tmp}"

# Lite-ApexNav defaults are bounded so each perception call concerns its current frame.
export VLM_REQUEST_ATTEMPTS="${VLM_REQUEST_ATTEMPTS:-1}"
export VLM_REQUEST_TIMEOUT="${VLM_REQUEST_TIMEOUT:-5}"
export VLM_RETRY_BACKOFF="${VLM_RETRY_BACKOFF:-0.5}"

if ! "$LITE_PYTHON" "$LITE_PREFLIGHT" "$LOCAL_HABITAT" "$PATCH_FILE"; then
    fail "Lite multi-agent runtime preflight failed. Check the Lite environment and run scripts/setup_lite_multiagent_habitat.sh."
fi

cd "${LITE_APEX_WORKDIR:-$APEXNAV_ROOT}"
exec "$LITE_PYTHON" "$@"
