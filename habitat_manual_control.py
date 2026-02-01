"""
Manual Habitat ObjectNav Runner (HM3D/MP3D) with LLM Integration
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
import numpy as np
import cv2
import rospy
from omegaconf import DictConfig  # 修复 NameError 的关键导入
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
FORWARD_KEY = "w"
LEFT_KEY = "a"
RIGHT_KEY = "d"
LOOK_UP_KEY = "q"
LOOK_DOWN_KEY = "e"
FINISH = "f"
fusion_threshold = 0.5

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
    print("\nManual controls:")
    print(f"  {FORWARD_KEY} - Move forward")
    print(f"  {LEFT_KEY} - Turn left")
    print(f"  {RIGHT_KEY} - Turn right")
    print(f"  {LOOK_UP_KEY} - Look up")
    print(f"  {LOOK_DOWN_KEY} - Look down")
    print(f"  {FINISH} - Stop (end episode)")
    print("  Ctrl+C - Quit (graceful shutdown)")
    print("Note: Focus the 'Observations' window before pressing keys.\n")

def publish_observations(event):
    global msg_observations, fusion_threshold
    global ros_pub, confidence_threshold_pub
    tmp = deepcopy(msg_observations)
    ros_pub.habitat_publish_ros_topic(tmp)
    publish_float64(confidence_threshold_pub, fusion_threshold)

def _parse_dataset_arg():
    parser = argparse.ArgumentParser(description="Habitat Manual Runner", add_help=True)
    parser.add_argument("--dataset", type=str, choices=["hm3dv1", "hm3dv2", "mp3d"], default="hm3dv2")
    args, unknown = parser.parse_known_args()
    return args.dataset, unknown

def main(cfg: DictConfig) -> None:
    global msg_observations, fusion_threshold
    global ros_pub, confidence_threshold_pub

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
    llm_response_path = llm_cfg.llm_response_path

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
    
    env = habitat.Env(cfg)
    print("Environment creation successful")

    while env_count:
        env.current_episode = next(env.episode_iterator)
        env_count -= 1
    
    observations = env.reset()
    observations["rgb"] = transform_rgb_bgr(observations["rgb"])
    info = env.get_metrics()
    frame = observations_to_image(observations, info)
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("Observations", small_frame)

    camera_pitch = 0.0
    observations["camera_pitch"] = camera_pitch
    msg_observations = deepcopy(observations)

    ros_pub = habitat_publisher.ROSPublisher()
    timer = rospy.Timer(rospy.Duration(0.1), publish_observations)
    itm_score_pub = rospy.Publisher("/blip2/cosine_score", Float64, queue_size=10)
    cld_with_score_pub = rospy.Publisher("/detector/clouds_with_scores", MultipleMasksWithConfidence, queue_size=10)
    confidence_threshold_pub = rospy.Publisher("/detector/confidence_threshold", Float64, queue_size=10)

    print("Agent stepping around inside environment.")
    print_manual_controls()

    label = env.current_episode.object_category
    if label in category_to_coco:
        coco_id = category_to_coco[label]
        label = id_to_name.get(coco_id, label)

    # --- LLM Answer Fetching and Robustness Fix ---
    llm_answer, room, fusion_threshold = read_answer(llm_answer_path, llm_response_path, label, llm_client)
    
    if llm_answer is None or not isinstance(llm_answer, list):
        print(f"Warning: Invalid llm_answer for {label}, using default.")
        llm_answer = []
    
    # 防止 index out of range：确保长度至少为 2
    while len(llm_answer) < 2:
        llm_answer.append("stop")
    # ----------------------------------------------

    cld_with_score_msg = MultipleMasksWithConfidence()
    count_steps = 0

    while not rospy.is_shutdown() and not env.episode_over:
        print(f"\n-------------Step: {count_steps}-------------")
        keystroke = cv2.waitKey(0)
        if keystroke == ord(FORWARD_KEY):
            action = HabitatSimActions.move_forward
        elif keystroke == ord(LOOK_UP_KEY):
            action = HabitatSimActions.look_up
            camera_pitch += np.pi / 6.0
        elif keystroke == ord(LOOK_DOWN_KEY):
            action = HabitatSimActions.look_down
            camera_pitch -= np.pi / 6.0
        elif keystroke == ord(LEFT_KEY):
            action = HabitatSimActions.turn_left
        elif keystroke == ord(RIGHT_KEY):
            action = HabitatSimActions.turn_right
        elif keystroke == ord(FINISH):
            action = HabitatSimActions.stop
        else:
            continue

        timer.shutdown()
        observations = env.step(action)
        count_steps += 1
        info = env.get_metrics()

        cosine = get_itm_message_cosine(observations["rgb"], label, room)
        publish_float64(itm_score_pub, cosine)

        # Double check llm_answer before calling get_object
        if not llm_answer:
            llm_answer = ["stop", "stop"]

        detect_img, score_list, object_masks_list, label_list = get_object(
            label, observations["rgb"], detector_cfg, llm_answer
        )

        observations["rgb"] = detect_img
        observations["camera_pitch"] = camera_pitch
        ros_pub.habitat_publish_ros_topic(observations)
        observations["rgb"] = transform_rgb_bgr(detect_img)
        render_obs = {
            k: v for k, v in observations.items() 
            if hasattr(v, "shape") and isinstance(v, (np.ndarray, list))
        }
        
        # 使用过滤后的字典进行可视化
        frame = observations_to_image(render_obs, info)

        obj_point_cloud_list = get_object_point_cloud(cfg, observations, object_masks_list)
        cld_with_score_msg.point_clouds = obj_point_cloud_list
        cld_with_score_msg.confidence_scores = score_list
        cld_with_score_msg.label_indices = label_list
        cld_with_score_pub.publish(cld_with_score_msg)

        cv2.imshow("Observations", frame)

    env.close()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("habitat_ros_publisher", anonymous=True)

    try:
        dataset, overrides = _parse_dataset_arg()
        cfg_name = f"habitat_eval_{dataset}"
        with initialize(version_base=None, config_path="config"):
            cfg = compose(config_name=cfg_name, overrides=overrides)
        main(cfg)
        
    except Exception as e:
        print("----- Detailed Error Traceback -----")
        traceback.print_exc() # 这会打印具体的行号和函数调用链
        print("------------------------------------")
        
        
        rospy.signal_shutdown("Shutdown due to error")
        os._exit(1)
