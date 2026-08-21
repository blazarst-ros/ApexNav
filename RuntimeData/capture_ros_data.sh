#!/usr/bin/env bash
# Record a bounded, self-describing ApexNav runtime evidence bundle.
set -Eeuo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: capture_ros_data.sh [planning|planning-full|mission|mapping|perception] [duration_sec]

  planning       Planner FSM, KinoAstar, MINCO, MPC, map snapshots and control loop (default)
  planning-full  planning plus synchronized RGB/depth/semantic data (large)
  mission        PX4, supervisor, gates, planner state and command output
  mapping        Camera pose/depth, occupancy, ESDF and object/value maps
  perception     RGB-D input, VLM diagnostics/observations and semantic map output

duration_sec=0 records until Ctrl-C. Example: ./capture_ros_data.sh planning 90
EOF
  exit 64
}

profile="${1:-planning}"
duration="${2:-0}"
[[ "$duration" =~ ^[0-9]+$ ]] || usage

case "$profile" in
  planning|planning-full|mission|mapping|perception) ;;
  *) usage ;;
esac

runtime_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace_dir="$(cd "${runtime_dir}/.." && pwd)"
# rostopic must know ApexNav's custom message classes even when this script is
# launched from a fresh terminal.
if [[ -f /opt/ros/noetic/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/noetic/setup.bash
fi
if [[ -f "${workspace_dir}/devel/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${workspace_dir}/devel/setup.bash"
fi
stamp="$(date +%Y%m%d_%H%M%S_%z)"
out_dir="${runtime_dir}/capture_${profile}_${stamp}"
mkdir -p "$out_dir"

common_topics=(
  /clock /tf /tf_static /rosout /rosout_agg
  /gazebo/model_states
  /apexnav/mission/state
  /apexnav/mission/mapping_enabled
  /apexnav/mission/navigation_enabled
  /apexnav/logging/state_log_file
)

mission_topics=(
  /mavros/state /mavros/extended_state /mavros/statustext/recv /mavros/battery
  /mavros/local_position/odom /mavros/setpoint_raw/local
  /mavros/vision_pose/pose
  /apexnav/sensors/diagnostics /apexnav/vlm/diagnostics
  /grid_map/commit
  /move_base_simple/goal /apexnav/planner/cancel /traj_server/stop
  /ros/state /ros/expl_state /ros/expl_result
  /planning/trajectory /apexnav/planner/cmd_vel_raw
  /apexnav/planner/trajectory_reference /apexnav/planner/trajectory_progress
)

planner_topics=(
  /detector/label /detector/confidence_threshold
  /apexnav/camera/pose /apexnav/vlm/semantic_observation
  /grid_map/commit /grid_map/filtered_depth_cloud
  /grid_map/occupied /grid_map/free /grid_map/unknown
  /grid_map/occupied_inflate /grid_map/esdf
  /grid_map/value_map /grid_map/occupancy_object /object/clouds
  /planning/trajectory
  /apexnav/planner/trajectory_reference /apexnav/planner/trajectory_progress
  /exploration_node/kinoastar/expanded_nodes
  /kinoastar/FlatPath /kinoastar/FlatTraj
  /trajectory/minco_init_path /trajectory/minco_init_path_alpha_pub_
  /trajectory/mincoPath /trajectory/minco_opt_path_alpha_pub_
  /trajectory/innerpoint /trajectory/initinnerpoint
  /planning_vis/trajectory /planning_vis/topo_path
  /planning_vis/prediction /planning_vis/visib_constraint
  /planning_vis/frontier /planning_vis/yaw /planning_vis/viewpoints
  /traj_server_node/mpc_car/predict_path
  /traj_server_node/mpc_car/reference_path
  /traj_server_node/mpc_car/track_err
  /current_desire /travel_traj /robot
)

mapping_topics=(
  /apexnav/camera/pose /apexnav/camera/camera_info
  /apexnav/camera/depth/image_raw
  /grid_map/commit /grid_map/depth_cloud /grid_map/filtered_depth_cloud
  /grid_map/occupied /grid_map/free /grid_map/unknown
  /grid_map/occupied_inflate /grid_map/esdf /grid_map/update_range
  /grid_map/value_map /grid_map/occupancy_object
  /grid_map/all_object_cloud /grid_map/filtered_object_cloud
  /grid_map/over_depth_object_cloud /object/clouds
)

perception_topics=(
  /detector/label /detector/confidence_threshold
  /apexnav/camera/rgb/image_raw /apexnav/camera/depth/image_raw
  /apexnav/camera/camera_info /apexnav/camera/pose
  /apexnav/vlm/diagnostics /apexnav/vlm/semantic_observation
  /apexnav/vlm/annotated_image /apexnav/vlm/status_marker
  /grid_map/all_object_cloud /grid_map/filtered_object_cloud
  /grid_map/over_depth_object_cloud /grid_map/occupancy_object /object/clouds
)

requested_topics=("${common_topics[@]}")
case "$profile" in
  planning)
    requested_topics+=("${mission_topics[@]}" "${planner_topics[@]}")
    ;;
  planning-full)
    requested_topics+=("${mission_topics[@]}" "${planner_topics[@]}" "${perception_topics[@]}")
    ;;
  mission)
    requested_topics+=("${mission_topics[@]}")
    ;;
  mapping)
    requested_topics+=("${mapping_topics[@]}")
    ;;
  perception)
    requested_topics+=("${perception_topics[@]}")
    ;;
esac

if ! rostopic list > "${out_dir}/topic_list.txt" 2> "${out_dir}/ros_master_error.txt"; then
  echo "Cannot contact ROS master. See ${out_dir}/ros_master_error.txt" >&2
  exit 69
fi

mapfile -t requested_topics < <(printf '%s\n' "${requested_topics[@]}" | sort -u)
available_topics=()
missing_topics=()
for topic in "${requested_topics[@]}"; do
  if grep -Fqx "$topic" "${out_dir}/topic_list.txt"; then
    available_topics+=("$topic")
  else
    missing_topics+=("$topic")
  fi
done

# Include future planner debug publishers without requiring script changes.
while IFS= read -r topic; do
  case "$topic" in
    /exploration_node/kinoastar/*|/kinoastar/*|/trajectory/*|/planning_vis/*|\
    /traj_server_node/mpc_car/*|/apexnav/planner/*)
      available_topics+=("$topic")
      ;;
  esac
done < "${out_dir}/topic_list.txt"
mapfile -t available_topics < <(printf '%s\n' "${available_topics[@]}" | sort -u)

if [[ ${#available_topics[@]} -eq 0 ]]; then
  echo "No requested topics are currently advertised." >&2
  exit 69
fi

printf '%s\n' "${available_topics[@]}" > "${out_dir}/recorded_topics.txt"
printf '%s\n' "${missing_topics[@]}" > "${out_dir}/missing_topics_at_start.txt"
printf '%s\n' \
  "profile=${profile}" \
  "started_at=$(date --iso-8601=seconds)" \
  "duration_sec=${duration}" \
  "ros_master_uri=${ROS_MASTER_URI:-http://localhost:11311}" \
  "hostname=$(hostname)" \
  "recorded_topic_count=${#available_topics[@]}" \
  "missing_topic_count=${#missing_topics[@]}" > "${out_dir}/capture_metadata.txt"

collect_static_evidence() {
  rostopic list -v > "${out_dir}/topic_list_verbose.txt" 2>&1 || true
  rosnode list > "${out_dir}/node_list.txt" 2>&1 || true
  rosnode info /exploration_node > "${out_dir}/exploration_node_info.txt" 2>&1 || true
  rosnode info /traj_server_node > "${out_dir}/traj_server_node_info.txt" 2>&1 || true
  rosnode info /px4_mission_supervisor \
    > "${out_dir}/mission_supervisor_node_info.txt" 2>&1 || true
  rosparam get /exploration_node > "${out_dir}/exploration_node_params.yaml" 2>&1 || true
  rosparam get /traj_server_node > "${out_dir}/traj_server_node_params.yaml" 2>&1 || true
  rosparam get /px4_mission_supervisor \
    > "${out_dir}/mission_supervisor_params.yaml" 2>&1 || true
  timeout 5s rostopic echo -n 1 /apexnav/mission/state \
    > "${out_dir}/mission_state_at_start.txt" 2>&1 || true
  timeout 5s rostopic echo -n 1 /ros/state \
    > "${out_dir}/planner_state_at_start.txt" 2>&1 || true
  timeout 5s rostopic echo -n 1 /ros/expl_result \
    > "${out_dir}/exploration_result_at_start.txt" 2>&1 || true
}

bag_prefix="${out_dir}/${profile}"
bag_pid=""
evidence_pid=""
stopping=0
finalized=0
capture_status="starting"

stop_capture() {
  if [[ "$stopping" -eq 1 ]]; then
    return
  fi
  stopping=1
  if [[ -n "$bag_pid" ]] && kill -0 "$bag_pid" 2>/dev/null; then
    kill -INT "$bag_pid" 2>/dev/null || true
    wait "$bag_pid" 2>/dev/null || true
  fi
}

finalize_capture() {
  if [[ "$finalized" -eq 1 ]]; then
    return
  fi
  finalized=1
  stop_capture
  if [[ -n "$evidence_pid" ]]; then
    wait "$evidence_pid" 2>/dev/null || true
  fi
  printf '%s\n' \
    "ended_at=$(date --iso-8601=seconds)" \
    "capture_status=${capture_status}" \
    "elapsed_wall_sec=$((SECONDS - started_seconds))" \
    >> "${out_dir}/capture_metadata.txt"
  while IFS= read -r bag; do
    rosbag info "$bag"
  done < <(find "$out_dir" -maxdepth 1 -type f -name '*.bag' -print | sort) \
    > "${out_dir}/rosbag_info.txt" 2>&1 || true
}

handle_signal() {
  capture_status="interrupted_by_signal"
  finalize_capture
  exit 130
}

started_seconds=$SECONDS
trap 'handle_signal' INT TERM
trap 'finalize_capture' EXIT

echo "Recording ${profile} planner evidence to ${out_dir}"
echo "Topics: ${#available_topics[@]}; missing at start: ${#missing_topics[@]}"
echo "Use Ctrl-C to stop. Large bags are split at 2048 MB."

rosbag record --lz4 --split --size=2048 --buffsize=1024 \
  --output-name="$bag_prefix" "${available_topics[@]}" \
  > "${out_dir}/rosbag_record.log" 2>&1 &
bag_pid=$!

# Static inspection starts only after rosbag is running, so it cannot delay capture.
collect_static_evidence &
evidence_pid=$!

capture_status="recording"
while kill -0 "$bag_pid" 2>/dev/null; do
  if (( duration > 0 && SECONDS - started_seconds >= duration )); then
    break
  fi
  sleep 1
done

if (( duration > 0 && SECONDS - started_seconds >= duration )); then
  capture_status="duration_complete"
else
  capture_status="rosbag_exited_early"
fi
finalize_capture
trap - EXIT INT TERM

echo "Capture complete: ${out_dir}"
echo "Inspect: ${out_dir}/rosbag_info.txt"
