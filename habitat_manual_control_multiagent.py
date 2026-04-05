"""
Multi-agent Manual Habitat ObjectNav Runner (HM3D/MP3D) with LLM Integration

Allows manual control of both agents simultaneously using keyboard:
  Agent 0: WASD + Q/E
  Agent 1: Arrow-like keys mapped to IJKL + U/O
"""

# Standard library imports
import argparse
import gzip
import json
import os
import traceback
import numpy as np
import signal
from copy import deepcopy

# Third-party library imports
from hydra import initialize, compose
import cv2
import rospy
from omegaconf import DictConfig
from std_msgs.msg import Float64

# Habitat-related imports
import habitat
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.visualizations.utils import (
    observations_to_image,
)

# ROS message imports
from plan_env.msg import MultipleMasksWithConfidence

# Local project imports
from habitat2ros import habitat_publisher
from vlm.utils.get_object_utils import get_object
from vlm.utils.get_itm_message import get_itm_message_cosine
from llm.answer_reader.answer_reader import read_answer
from basic_utils.object_point_cloud_utils.object_point_cloud import (
    get_object_point_cloud,
)
from vlm.Labels import MP3D_ID_TO_NAME

# Global settings
num_agents = 2  # Number of agents
AGENT_CHARS = {
    0: {"forward": "w", "left": "a", "right": "d", "up": "q", "down": "e", "finish": "f"},
    1: {"forward": "i", "left": "j", "right": "l", "up": "u", "down": "o", "finish": "k"},
}
fusion_threshold = 0.4

def signal_handler(sig, frame):
    print("Ctrl+C detected! Shutting down...")
    rospy.signal_shutdown("Manual shutdown")
    os._exit(0)

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def publish_float64(publisher, data: float):
    msg = Float64()
    msg.data = data
    publisher.publish(msg)

def print_manual_controls():
    print("\nMulti-agent manual controls:")
    for idx, keys in AGENT_CHARS.items():
        print(f"  Agent {idx}:")
        print(f"    {keys['forward']} - Move forward    {keys['left']}/{keys['right']} - Turn left/right")
        print(f"    {keys['up']} - Look up    {keys['down']} - Look down   {keys['finish']} - Stop")
    print("  Ctrl+C - Quit (graceful shutdown)")
    print("Note: Focus the 'Observations' window before pressing keys.\n")

def publish_observations(event):
    global msg_observations, fusion_threshold
    global ros_pub, confidence_threshold_pub
    tmp = deepcopy(msg_observations)
    ros_pub.habitat_publish_ros_topic(tmp)
    publish_float64(confidence_threshold_pub, fusion_threshold)

def _parse_dataset_arg():
    parser = argparse.ArgumentParser(description="Habitat Multi-Agent Manual Runner", add_help=True)
    parser.add_argument("--dataset", type=str, choices=["hm3dv1", "hm3dv2", "mp3d", "hm3dv2_multiagent", "mp3d_multiagent"], default="hm3dv2_multiagent")
    parser.add_argument("--num_agents", type=int, default=2, help="Number of agents")
    args, unknown = parser.parse_known_args()
    return args.dataset, unknown

def main(cfg: DictConfig) -> None:
    global msg_observations, fusion_threshold
    global ros_pub, confidence_threshold_pub

    num_agents = cfg.get("num_agents", 2)
    agent_names = [f"agent_{i}" for i in range(num_agents)]

    # Load dataset info
    with gzip.open("data/datasets/objectnav/mp3d/v1/val/val.json.gz", "rt", encoding="utf-8") as f:
        val_data = json.load(f)
    category_to_coco = val_data.get("category_to_mp3d_category_id", {})
    id_to_name = {category_to_coco[cat]: MP3D_ID_TO_NAME[idx] for idx, cat in enumerate(category_to_coco)}

    cfg = patch_config(cfg)
    env_count = 0 if cfg.test_epi_num == -1 else cfg.test_epi_num
    detector_cfg = cfg.detector
    llm_cfg = cfg.llm
    llm_client = llm_cfg.llm_client
    llm_answer_path = llm_cfg.llm_answer_path
    llm_response_path = cfg.llm.llm_response_path

    os.makedirs(os.path.dirname(llm_answer_path), exist_ok=True)

    # Config Habitat Measurements
    with habitat.config.read_write(cfg):
        cfg.habitat.task.measurements.update({
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3, map_resolution=256, draw_source=True, draw_border=True,
                draw_shortest_path=True, draw_view_points=True, draw_goal_positions=True,
                draw_goal_aabbs=False, fog_of_war=FogOfWarConfig(draw=True, visibility_dist=5.0, fov=79)
            ),
            "collisions": CollisionsMeasurementConfig(),
        })

    env = habitat.MultiAgentEnv(cfg)
    print(f"Multi-agent environment created with {num_agents} agents")

    while env_count:
        env.current_episode = next(env.episode_iterator)
        env_count -= 1

    observations = env.reset()

    # Setup per-agent publishers and state
    ros_pubs = {}
    for agent_name in agent_names:
        ros_pubs[agent_name] = habitat_publisher.ROSPublisher(agent_name)

    agent_states = {}
    for agent_name in agent_names:
        agent_obs = observations[agent_name]
        agent_obs["rgb"] = transform_rgb_bgr(agent_obs["rgb"])
        agent_obs["camera_pitch"] = 0.0

        info = env.get_metrics()
        agent_info = info.get(agent_name, info)
        frame = observations_to_image(agent_obs, agent_info)
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        agent_states[agent_name] = {
            "camera_pitch": 0.0,
            "count_steps": 0,
            "observations": agent_obs,
        }

    # Use agent_0 as primary for shared messages
    ros_pub = ros_pubs[agent_names[0]]
    timer = rospy.Timer(rospy.Duration(0.1), publish_observations)
    confidence_threshold_pub = rospy.Publisher("/detector/confidence_threshold", Float64, queue_size=10)

    # Per-agent ITM and cloud publishers
    _itm_pubs = {}
    _cld_pubs = {}

    print("Multi-agent stepping inside environment.")
    print_manual_controls()

    label = env.current_episode.object_category
    if label in category_to_coco:
        coco_id = category_to_coco[label]
        label = id_to_name.get(coco_id, label)

    llm_answer, room, fusion_threshold = read_answer(llm_answer_path, llm_response_path, label, llm_client)

    if llm_answer is None or not isinstance(llm_answer, list):
        print(f"Warning: Invalid llm_answer for {label}, using default.")
        llm_answer = []

    while len(llm_answer) < 2:
        llm_answer.append("stop")

    count_steps = 0

    while not rospy.is_shutdown():
        print(f"\n-------------Step: {count_steps}-------------")
        keystroke = cv2.waitKey(0)

        # Determine which agent is controlled and what action
        agent_idx = -1
        action_key = None
        for idx in range(num_agents):
            for key_name, key_char in AGENT_CHARS.get(idx, {}).items():
                if keystroke == ord(key_char):
                    agent_idx = idx
                    action_key = key_name
                    break
            if agent_idx >= 0:
                break

        if agent_idx < 0:
            continue

        agent_name = f"agent_{agent_idx}"
        ast = agent_states[agent_name]
        action = None

        cfg_keys = AGENT_CHARS[agent_idx]
        if action_key == "forward":
            action = HabitatSimActions.move_forward
        elif action_key == "up":
            action = HabitatSimActions.look_up
            ast["camera_pitch"] += np.pi / 6.0
        elif action_key == "down":
            action = HabitatSimActions.look_down
            ast["camera_pitch"] -= np.pi / 6.0
        elif action_key == "left":
            action = HabitatSimActions.turn_left
        elif action_key == "right":
            action = HabitatSimActions.turn_right
        elif action_key == "finish":
            action = HabitatSimActions.stop
        else:
            continue

        timer.shutdown()
        observations = env.step({agent_name: action})
        count_steps += 1
        info = env.get_metrics()

        agent_obs = observations[agent_name]
        cosine = get_itm_message_cosine(agent_obs["rgb"], label, room)

        # Per-agent ITM score publisher
        itm_topic = f"/blip2/{agent_name}/cosine_score"
        if itm_topic not in _itm_pubs:
            _itm_pubs[itm_topic] = rospy.Publisher(itm_topic, Float64, queue_size=10)
        _itm_pubs[itm_topic].publish(Float64(cosine))

        if not llm_answer:
            llm_answer = ["stop", "stop"]

        detect_img, score_list, object_masks_list, label_list = get_object(
            label, agent_obs["rgb"], detector_cfg, llm_answer
        )

        agent_obs["rgb"] = detect_img
        agent_obs["camera_pitch"] = ast["camera_pitch"]
        ros_pubs[agent_name].habitat_publish_ros_topic(agent_obs)

        # Per-agent point cloud publisher
        from basic_utils.object_point_cloud_utils.object_point_cloud import get_object_point_cloud
        obj_point_cloud_list = get_object_point_cloud(
            cfg, {**agent_obs}, object_masks_list, agent_name
        )
        cld_topic = f"/detector/{agent_name}/clouds_with_scores"
        if cld_topic not in _cld_pubs:
            _cld_pubs[cld_topic] = rospy.Publisher(cld_topic, MultipleMasksWithConfidence, queue_size=10)
        cld_msg = MultipleMasksWithConfidence()
        cld_msg.point_clouds = obj_point_cloud_list
        cld_msg.confidence_scores = score_list
        cld_msg.label_indices = label_list
        _cld_pubs[cld_topic].publish(cld_msg)

        render_obs = {
            k: v for k, v in agent_obs.items()
            if hasattr(v, "shape") and isinstance(v, (np.ndarray, list))
        }
        agent_info = info.get(agent_name, info)
        frame = observations_to_image(render_obs, agent_info)
        cv2.imshow(f"Agent {agent_idx}", frame)

    env.close()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("habitat_multiagent_publisher", anonymous=True)

    try:
        dataset, overrides = _parse_dataset_arg()
        cfg_name = f"habitat_eval_{dataset}"
        with initialize(version_base=None, config_path="config"):
            cfg = compose(config_name=cfg_name, overrides=overrides)
        main(cfg)

    except Exception as e:
        print("----- Detailed Error Traceback -----")
        traceback.print_exc()
        print("------------------------------------")

        rospy.signal_shutdown("Shutdown due to error")
        os._exit(1)
