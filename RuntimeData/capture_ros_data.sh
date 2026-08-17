#!/usr/bin/env bash
# Capture one evidence bundle for a single ApexNav data-flow stage.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {recognition|object_filter|depth_mapping|shared_map|planning|pitch}" >&2
  exit 64
fi

stage="$1"
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
stamp="$(date +%Y%m%d_%H%M%S)"
out_dir="${root_dir}/capture_${stage}_${stamp}"
mkdir -p "$out_dir"

agents=(0 1)
topics=()

add_agent_topics() {
  local suffix
  suffix="$1"
  for agent in "${agents[@]}"; do
    topics+=("/habitat/agent_${agent}/${suffix}")
  done
}

case "$stage" in
  recognition)
    add_agent_topics camera_rgb
    add_agent_topics camera_depth
    for agent in "${agents[@]}"; do
      topics+=("/detector/agent_${agent}/clouds_with_scores")
      topics+=("/clip/agent_${agent}/cosine_score")
    done
    topics+=(/stage1/detector/detection /detector/confidence_threshold)
    ;;
  object_filter)
    for agent in "${agents[@]}"; do
      topics+=("/detector/agent_${agent}/clouds_with_scores")
    done
    topics+=(/grid_map/all_object_cloud /grid_map/filtered_object_cloud)
    topics+=(/grid_map/over_depth_object_cloud /grid_map/semantic_objects)
    topics+=(/grid_map/occupancy_object /object/cluster_status /object/cluster_markers)
    ;;
  pitch)
    add_agent_topics sensor_pose
    for agent in "${agents[@]}"; do
      topics+=("/map_ros/agent_${agent}/camera_pitch")
      topics+=("/detector/agent_${agent}/clouds_with_scores")
    done
    topics+=(/grid_map/all_object_cloud /grid_map/filtered_object_cloud)
    topics+=(/object/cluster_status)
    ;;
  depth_mapping)
    add_agent_topics sensor_pose
    add_agent_topics camera_depth
    topics+=(/grid_map/depth_cloud /grid_map/filtered_depth_cloud)
    topics+=(/grid_map/occupied /grid_map/free /grid_map/unknown)
    topics+=(/grid_map/occupied_inflate /grid_map/esdf)
    ;;
  shared_map)
    for agent in "${agents[@]}"; do
      topics+=("/detector/agent_${agent}/clouds_with_scores")
      topics+=("/clip/agent_${agent}/cosine_score")
    done
    topics+=(/grid_map/value_map /grid_map/semantic_objects /grid_map/occupancy_object)
    topics+=(/object/cluster_status /object/cluster_markers /grid_map/occupied /grid_map/esdf)
    ;;
  planning)
    add_agent_topics odom
    for agent in "${agents[@]}"; do
      topics+=("/ros/agent_${agent}/exploration_strategy")
      topics+=("/habitat/plan_action_agent_${agent}")
      topics+=("/ros/agent_${agent}/expl_result")
      topics+=("/robot_agent_${agent}")
    done
    topics+=(/habitat/state /ros/state_all /ros/expl_result_all)
    topics+=(/grid_map/occupied_inflate /grid_map/esdf /grid_map/value_map /object/cluster_status)
    ;;
  *)
    echo "Unknown stage: $stage" >&2
    exit 64
    ;;
esac

printf '%s\n' "stage=${stage}" "started_at=$(date --iso-8601=seconds)" > "${out_dir}/capture_metadata.txt"
printf '%s\n' "${topics[@]}" > "${out_dir}/topic_manifest.txt"
rosnode info /exploration_node > "${out_dir}/exploration_node_info.txt" 2>&1 || true
rostopic list | sort > "${out_dir}/topic_list.txt" 2>&1 || true

for topic in "${topics[@]}"; do
  safe_name="${topic#/}"
  safe_name="${safe_name//\//_}"
  rostopic info "$topic" > "${out_dir}/${safe_name}.info.txt" 2>&1 || true
  timeout 10s rostopic hz "$topic" > "${out_dir}/${safe_name}.hz.txt" 2>&1 || true
done

echo "Capturing ${stage} data in ${out_dir}. Press Ctrl-C after sufficient motion/detections."
rosbag record --output-name="${out_dir}/${stage}" "${topics[@]}"
