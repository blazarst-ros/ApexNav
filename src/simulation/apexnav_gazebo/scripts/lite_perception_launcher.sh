#!/usr/bin/env bash
set -Eeuo pipefail

PACKAGE_DIR="$(rospack find apexnav_gazebo)"
DEFAULT_REPO_DIR="$(cd "${PACKAGE_DIR}/../../.." && pwd)"
REPO_DIR="${APEXNAV_REPO_DIR:-${DEFAULT_REPO_DIR}}"
LITE_ENV="${APEXNAV_LITE_ENV:-/media/blazarst/Getea/Lite-ApexNav/conda-env/Lite-apex}"
LITE_PYTHON="${LITE_ENV}/bin/python"
SYSTEM_LIBFFI="/usr/lib/x86_64-linux-gnu/libffi.so.7"

[[ -x "${LITE_PYTHON}" ]] || {
  echo "Lite Python not found: ${LITE_PYTHON}" >&2
  exit 2
}
[[ -f "${REPO_DIR}/real_world_test_example/real_world_perception.py" ]] || {
  echo "ApexNav perception source not found under ${REPO_DIR}" >&2
  exit 2
}

# ROS Noetic's cv_bridge links against the system libp11-kit/libffi ABI, while
# the Lite conda OpenCV stack ships libffi.so.8.  Preload the compatible system
# ABI only for the perception process; VLM servers remain pure Lite Python.
if [[ -f "${SYSTEM_LIBFFI}" ]]; then
  export LD_PRELOAD="${SYSTEM_LIBFFI}${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

exec "${LITE_PYTHON}" "${REPO_DIR}/real_world_test_example/real_world_perception.py" "$@"
