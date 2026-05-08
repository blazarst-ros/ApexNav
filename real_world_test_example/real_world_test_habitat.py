#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import rospy
import numpy as np
import time
from cv_bridge import CvBridge
import message_filters
import tf.transformations as tft

import hydra
from omegaconf import DictConfig

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, String
from plan_env.msg import MultipleMasksWithConfidence
#自定义 ROS 消息MultipleMasksWithConfidence（包含点云、置信度、标签索引），/detector/clouds_with_scores
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from vlm.utils.get_object_utils import get_object
from vlm.utils.get_itm_message import get_itm_message_cosine
from llm.answer_reader.answer_reader import read_answer
from llm.answer_reader.semantic_prior_ros import compute_mask_scales, publish_semantic_priors_to_ros
from basic_utils.object_point_cloud_utils.object_point_cloud import (
    get_object_point_cloud,
)
"""
real_world_node 节点（Multi-agent 双版本）
核心任务：
为每个 agent 并行运行独立的感知流水线（检测 + ITM + 点云），
将结果发布到各自代理命名空间的话题，供 C++ planner 消费。
"""


def _get_num_agents(cfg):
    """Read NUM_AGENTS from config with fallback to 2."""
    num = getattr(cfg, "num_agents", 2) if cfg else 2
    return num if num else 2

def inverse_habitat_publisher_transform(sensor_pose_msg):
    """
    Inverse transform to recover original Habitat gps and compass from ROS sensor_pose.
    """
    pos = sensor_pose_msg.pose.pose.position
    orn = sensor_pose_msg.pose.pose.orientation

    # Invert position transform:
    gps = np.array([-pos.y, pos.z - 0.88, -pos.x], dtype=np.float32)

    # Invert orientation transform:
    euler = tft.euler_from_quaternion([orn.x, orn.y, orn.z, orn.w])
    compass_scalar = euler[2] + np.pi / 2.0
    # Habitat compass is a single-element array
    compass = np.array([compass_scalar], dtype=np.float32)

    return gps, compass


class AgentPerceptionPipeline:
    """Per-agent perception pipeline: sync RGB/depth/pose → detect + ITM → publish."""

    def __init__(self, cfg, agent_name, shared_state):
        self.config = cfg
        self.agent_name = agent_name
        self.shared = shared_state
        self.bridge = CvBridge()

        # Agent-namespaced subscribers
        self.rgb_sub_ = message_filters.Subscriber(f"/habitat/{agent_name}/camera_rgb", Image)
        self.depth_sub_ = message_filters.Subscriber(f"/habitat/{agent_name}/camera_depth", Image)
        self.sensor_pose_sub_ = message_filters.Subscriber(
            f"/habitat/{agent_name}/sensor_pose", Odometry
        )

        rospy.Subscriber(f"/habitat/{agent_name}/odom", Odometry, self.odom_callback, queue_size=10)

        # Per-agent publishers (namespace matches convention used by map_ros.cpp)
        self.itm_score_pub_ = rospy.Publisher(
            f"/blip2/{agent_name}/cosine_score", Float64, queue_size=10
        )
        self.cld_with_score_pub_ = rospy.Publisher(
            f"/detector/{agent_name}/clouds_with_scores", MultipleMasksWithConfidence, queue_size=10
        )

        # Synchronized callbacks: detection + value
        self.sync_detect = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub_, self.depth_sub_, self.sensor_pose_sub_],
            queue_size=5, slop=0.01,
        )
        self.sync_detect.registerCallback(self.sync_detect_callback)

        self.sync_value = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub_, self.depth_sub_, self.sensor_pose_sub_],
            queue_size=5, slop=0.01,
        )
        self.sync_value.registerCallback(self.sync_value_callback)

        # Per-agent state
        self.robot_odom = None
        self.processing_detect = False
        self.processing_value = False

        # Shared LLM config (read once by owner)
        self.llm_answer_path = self.shared.llm_answer_path
        self.llm_response_path = self.shared.llm_response_path
        self.llm_client_cfg = self.shared.llm_client_cfg

    def sync_detect_callback(self, rgb_msg, depth_msg, sensor_pose_msg):
        if self.processing_detect:
            return
        self.processing_detect = True
        try:
            stamp = rgb_msg.header.stamp
            time_diff = abs((stamp - sensor_pose_msg.header.stamp).to_sec())
            if time_diff > 0.1:
                return

            rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            transform_depth_img = depth_img.astype(np.float32)
            depth_cv = np.expand_dims(transform_depth_img, axis=-1)

            # Read label/LLM from shared state
            self.label = self.shared.label
            self.llm_answer = self.shared.llm_answer
            self.room = self.shared.room

            cld_with_score_msg = MultipleMasksWithConfidence()
            cld_with_score_msg.point_clouds = []
            cld_with_score_msg.confidence_scores = []
            cld_with_score_msg.label_indices = []
            cld_with_score_msg.mask_scales = []
            rospy.loginfo("detect: [%s] label: %s", self.agent_name, self.label)

            if self.label is None:
                rospy.logwarn_throttle(5.0, "[%s] Waiting for target label", self.agent_name)
                return

            detect_img, score_list, object_masks_list, label_list = get_object(
                self.label, rgb_cv, self.config.detector, self.llm_answer
            )

            gps, compass = inverse_habitat_publisher_transform(sensor_pose_msg)

            observations = {
                "depth": depth_cv,
                "gps": gps,
                "compass": compass,
            }

            obj_point_cloud_list = get_object_point_cloud(
                self.config, observations, object_masks_list, self.agent_name
            )
            cld_with_score_msg.point_clouds = obj_point_cloud_list
            cld_with_score_msg.confidence_scores = score_list
            cld_with_score_msg.label_indices = label_list
            cld_with_score_msg.mask_scales = compute_mask_scales(object_masks_list)

            self.cld_with_score_pub_.publish(cld_with_score_msg)
        except Exception as e:
            rospy.logerr("[%s] detect: Error in synchronized processing: %s", self.agent_name, e)
        finally:
            self.processing_detect = False

    def sync_value_callback(self, rgb_msg, depth_msg, sensor_pose_msg):
        if self.processing_value:
            return
        self.processing_value = True
        try:
            stamp = rgb_msg.header.stamp
            time_diff = abs((stamp - sensor_pose_msg.header.stamp).to_sec())
            if time_diff > 0.1:
                return

            rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")

            self.label = self.shared.label
            self.room = self.shared.room

            cosine = get_itm_message_cosine(rgb_cv, self.label, self.room)
            rospy.loginfo("value: [%s] cosine score: %.3f", self.agent_name, cosine)

            itm_score_msg = Float64()
            itm_score_msg.data = cosine
            self.itm_score_pub_.publish(itm_score_msg)

        except Exception as e:
            rospy.logerr("[%s] value: Error in synchronized processing: %s", self.agent_name, e)
        finally:
            self.processing_value = False

    def label_callback(self, msg):
        """Delegate to shared state."""
        try:
            new_label = str(msg.data)
            if new_label == self.shared.label:
                return
            self.shared.label = new_label
            rospy.loginfo("Received target label: %s", self.shared.label)
            try:
                self.shared.llm_answer, self.shared.room, self.shared.fusion_score = read_answer(
                    self.llm_answer_path, self.llm_response_path, self.shared.label, self.llm_client_cfg
                )
                publish_semantic_priors_to_ros(
                    self.llm_answer_path, self.shared.label, self.shared.llm_answer
                )
            except Exception:
                self.shared.llm_answer = []
                self.shared.room = None
                self.shared.fusion_score = 0.0
        except Exception as e:
            rospy.logerr("label_callback: Error processing label message: %s", e)

    def odom_callback(self, msg):
        try:
            self.robot_odom = msg
            self.odom_stamp = msg.header.stamp
            if self.odom_stamp is not None:
                self.odom_stamp = None
        except Exception as e:
            rospy.logerr("[%s] odom: Error processing Odometry: %s", self.agent_name, e)


class SharedLLMState:
    """Shared LLM configuration and label state across all agent pipelines."""

    def __init__(self, cfg):
        llm_cfg = cfg.llm
        self.llm_answer_path = llm_cfg.llm_answer_path
        self.llm_response_path = llm_cfg.llm_response_path
        self.llm_client_cfg = llm_cfg.llm_client

        self.label = None
        self.llm_answer = []
        self.room = None
        self.fusion_score = 0.0


class MultiAgentNode:
    def __init__(self, cfg):
        self.config = cfg

        rospy.init_node("habitat_multiagent_perception", anonymous=False)

        # Determine number of agents from config
        self.num_agents = _get_num_agents(cfg)

        # Shared state: LLM config + label
        self.shared = SharedLLMState(cfg)

        # Shared confidence threshold publisher (one for all agents)
        self.confidence_threshold_pub_ = rospy.Publisher(
            "/detector/confidence_threshold", Float64, queue_size=10
        )
        rospy.Timer(rospy.Duration(1.0), self.publish_confidence_threshold)

        # Subscribe to label topic (single source of truth — shared across agents)
        rospy.Subscriber("/detector/label", String, self._on_label, queue_size=1)

        # Create per-agent pipelines
        agent_names = [f"agent_{i}" for i in range(self.num_agents)]
        self.pipelines = {}
        for name in agent_names:
            pipeline = AgentPerceptionPipeline(cfg, name, self.shared)
            self.pipelines[name] = pipeline

        rospy.loginfo(f"Multi-agent perception node created with {self.num_agents} agents")

    def _on_label(self, msg):
        """Forward label to all pipelines."""
        for p in self.pipelines.values():
            p.label_callback(msg)

    def publish_confidence_threshold(self, event):
        msg = Float64()
        msg.data = 0.4
        self.confidence_threshold_pub_.publish(msg)

    def run(self):
        rospy.loginfo(f"Multi-agent perception node running with {self.num_agents} agents.")
        rospy.spin()


@hydra.main(version_base=None, config_path="config", config_name="real_world_test")
def main(cfg: DictConfig):
    node = MultiAgentNode(cfg)
    node.run()


if __name__ == "__main__":
    main()
