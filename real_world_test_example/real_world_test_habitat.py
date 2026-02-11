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
from basic_utils.object_point_cloud_utils.object_point_cloud import (
    get_object_point_cloud,
)
"""
real_world_node 节点
核心任务：
初始化 Habitat 仿真环境 / 真实机器人的视觉、位姿输入；
订阅目标标签 ROS 话题，触发 LLM 语义信息提取；
调用目标检测模型（GroundingDINO/YOLOv7），结合 LLM 语义优化检测结果；
发布带语义信息的检测结果 ROS 话题，供 C++ 执行层读取；
处理轨迹跟踪、MPC 控制等运动指令，完成 “感知→决策→执行” 的闭环。
"""

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


class RealWorldNode:
    def __init__(self, cfg):
        self.config = cfg

        rospy.init_node("real_world_node", anonymous=False)

        self.bridge = CvBridge()

        # Configure subscribers（传感器订阅（RGB/深度/位姿））
        self.rgb_sub_ = message_filters.Subscriber("/habitat/camera_rgb", Image)
        self.depth_sub_ = message_filters.Subscriber("/habitat/camera_depth", Image)
        self.sensor_pose_sub_ = message_filters.Subscriber(
            "/habitat/sensor_pose", Odometry
        )

        rospy.Subscriber("/habitat/odom", Odometry, self.odom_callback, queue_size=10)


        # Configure publishers-----LLM相关ROS发布器（传递到执行层的核心通道）

        self.confidence_threshold_pub_ = rospy.Publisher(
            "/detector/confidence_threshold", Float64, queue_size=10
        )# 1. 发布LLM关联的置信度阈值（执行层目标检测过滤用）
        self.itm_score_pub_ = rospy.Publisher(
            "/blip2/cosine_score", Float64, queue_size=10
        )# 2. 发布LLM语义匹配的ITM余弦分数（执行层语义价值计算用）
        self.cld_with_score_pub_ = rospy.Publisher(
            "/detector/clouds_with_scores", MultipleMasksWithConfidence, queue_size=10
        )# 3. 发布带LLM语义的目标点云+置信度（执行层路径规划用）
        self.detect_img_pub_ = rospy.Publisher(
            "/detector/detect_img", Image, queue_size=10
        )# 4. 发布检测可视化图（含LLM语义过滤结果）


        # Initialize detector---传感器消息同步器
        # Synchronize RGB, depth and sensor_pose topics
        self.sync_detect = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub_, self.depth_sub_, self.sensor_pose_sub_],
            queue_size=5,
            slop=0.01,
        )
        self.sync_detect.registerCallback(self.sync_detect_callback)

        # Initialize value module
        # (uses synchronized RGB/depth/sensor_pose messages)
        self.sync_value = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub_, self.depth_sub_, self.sensor_pose_sub_],
            queue_size=5,
            slop=0.01,
        )
        self.sync_value.registerCallback(self.sync_value_callback)

        # Initialize odometry handling-----里程计初始化
        self.robot_odom = None
        self.T_base_camera = None
        self.odom_stamp = None
        # Processing flags: ensure we don't start a new processing run
        # until the previous one finished (rate adapts to available compute)
        self.processing_detect = False
        self.processing_value = False

        # LLM config (used when label is provided via topic)
        # ========== 核心：LLM配置初始化（从Hydra配置读取） ==========
        llm_cfg = self.config.llm
        self.llm_answer_path = llm_cfg.llm_answer_path# LLM答案缓存文件路径（如llm_answer_hm3d.txt）
        self.llm_response_path = llm_cfg.llm_response_path #LLM原始响应文件路径
        self.llm_client_cfg = llm_cfg.llm_client # LLM客户端（deepseek/ollama）

        # Label will be provided via ROS topic `/detector/label` (std_msgs/String)
        # Initialize empty/defaults; actual values will be set in `label_callback`.

        self.label = None   # 目标物体标签（如"chair"，从/detector/label话题接收）
        self.llm_answer = [] # LLM解析后的结构化列表（[误检测标签, 置信度, 房间]）
        self.room = None     # LLM提取的目标所属房间（如"living room"）
        self.fusion_score = 0.0   # LLM提取的融合置信度

        # 订阅目标标签话题，触发LLM内容提取（如：cabinet）
        rospy.Subscriber("/detector/label", String, self.label_callback, queue_size=1)

        rospy.Timer(rospy.Duration(1.0), self.publish_confidence_threshold)#周期性发布置信度阈值（LLM关联的检测阈值）


    def sync_detect_callback(self, rgb_msg, depth_msg, sensor_pose_msg):#LLM 信息融入目标检测，传递到执行层
        """
        校验传感器数据时间同步性；
        转换 ROS 图像格式为 OpenCV 可处理格式；

        结合 LLM 语义信息（llm_answer）执行目标检测；-----/home/blazarst/ApexNav/vlm/utils/get_object_utils.py 进行
        提取目标点云并封装为 ROS 消息；
        
        发布检测结果（可视化图 + 带置信度的点云）到执行层；
"""
        # If a detect run is already in progress, skip this invocation.
        if self.processing_detect:
            return
        self.processing_detect = True
        try:
            # rospy.loginfo("detect: Received synchronized RGB and depth images")
            stamp = rgb_msg.header.stamp
            time_diff = abs((stamp - sensor_pose_msg.header.stamp).to_sec())
            if time_diff > 0.1:
                # If timestamps differ significantly, skip this pair
                # and allow the next synchronized callback to run.
                return

            rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            depth_img = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding="passthrough"
            )
            transform_depth_img = depth_img.astype(np.float32)
            depth_cv = np.expand_dims(transform_depth_img, axis=-1)

            cld_with_score_msg = MultipleMasksWithConfidence()
            cld_with_score_msg.point_clouds = []
            cld_with_score_msg.confidence_scores = []
            cld_with_score_msg.label_indices = []
            rospy.loginfo("detect: label: %s", self.label)
            # rospy.loginfo("detect: room: %s", self.room)

            # If label not yet received, skip detection until available
            if self.label is None:
                rospy.logwarn_throttle(5.0, "Waiting for target label on /detector/label")
                return

            # ========== 核心LLM关联逻辑：LLM语义融入目标检测 ==========
            #基于LLM的llm_answer（误检测标签）过滤相似物体，提升目标检测精度
            detect_img, score_list, object_masks_list, label_list = get_object(
                self.label, rgb_cv, self.config.detector, self.llm_answer
            )

            # Use inverse transform to recover original Habitat observations format
            gps, compass = inverse_habitat_publisher_transform(sensor_pose_msg)

            observations = {
                "depth": depth_cv,
                "gps": gps,
                "compass": compass,  # Already a numpy array from inverse function
            }

            obj_point_cloud_list = get_object_point_cloud(
                self.config, observations, object_masks_list
            )
            cld_with_score_msg.point_clouds = obj_point_cloud_list
            cld_with_score_msg.confidence_scores = score_list
            cld_with_score_msg.label_indices = label_list
            # Publish the detection image for visualization
            self.detect_img_pub_.publish(
                self.bridge.cv2_to_imgmsg(detect_img, encoding="rgb8")
            )

            # Also publish the detected object clouds with scores so other nodes / RViz can use them
            self.cld_with_score_pub_.publish(cld_with_score_msg)
        except Exception as e:
            rospy.logerr("detect: Error in synchronized processing: %s", e)
        finally:
            # mark processing complete so next invocation can proceed
            self.processing_detect = False

    def sync_value_callback(self, rgb_msg, depth_msg, sensor_pose_msg):
        """
        语义价值计算回调函数，
        核心作用是结合 LLM 提取的 “目标标签 + 所属房间” 语义信息，
        计算当前场景与目标语义的匹配度（余弦分数），并将该分数发布到 ROS 话题
        """

        # If a value run is already in progress, skip this invocation.
        if self.processing_value:
            return
        self.processing_value = True
        try:
            stamp = rgb_msg.header.stamp
            time_diff = abs((stamp - sensor_pose_msg.header.stamp).to_sec())
            if time_diff > 0.1:
                # If timestamps differ significantly, skip this pair
                return

            rgb_cv = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
            # rospy.loginfo("value: room: %s", self.room)

            # ========== 核心LLM关联逻辑：计算语义匹配分数 ==========
            # 函数作用：计算当前场景与“目标标签+所属房间”的语义匹配度（余弦相似度，0~1）
            # 输出：cosine值越高，当前场景越可能包含目标物体
            cosine = get_itm_message_cosine(rgb_cv, self.label, self.room)
            rospy.loginfo("value: Computed cosine score: %.3f", cosine)

             # ========== 核心LLM关联逻辑：发布语义匹配分数到执行层 ==========
            itm_score_msg = Float64()
            itm_score_msg.data = cosine
            self.itm_score_pub_.publish(itm_score_msg)

        except Exception as e:
            rospy.logerr("value: Error in synchronized processing: %s", e)
        finally:
            self.processing_value = False

    def label_callback(self, msg):
        """
        ApexNav 感知层中LLM 语义信息提取的唯一触发函数，
        核心作用是接收外部发布的目标物体标签，
        触发 LLM 内容提取流程，
        并将提取到的 LLM 语义信息（误检测标签、目标房间、融合置信度）保存为类变量，
        供后续检测 / 语义匹配函数使用。
        """
        try:
            new_label = str(msg.data)
            if new_label == self.label:
                return
            self.label = new_label
            rospy.loginfo("Received target label: %s", self.label)
            # If LLM is configured, fetch LLM answer for the new label
            # ========== 核心LLM逻辑：提取LLM语义信息 ==========
            # 调用read_answer函数，根据新标签获取LLM内容：
            #   输入：LLM缓存文件路径、原始响应路径、目标标签、LLM客户端类型（deepseek/ollama）
            #   输出：llm_answer（误检测标签列表）、room（目标所属房间）、fusion_score（融合置信度）
            try:
                self.llm_answer, self.room, self.fusion_score = read_answer(
                  self.llm_answer_path, self.llm_response_path, self.label, self.llm_client_cfg # 现在传配置对象
                )
            except Exception:
                # Non-fatal: proceed without LLM answer
                self.llm_answer = []
                self.room = None
                self.fusion_score = 0.0
        except Exception as e:
            rospy.logerr("label_callback: Error processing label message: %s", e)

    def odom_callback(self, msg):
        try:
            self.robot_odom = msg
            self.odom_stamp = msg.header.stamp
            if self.odom_stamp is not None:
                # self.publish_sensor_pose()
                self.odom_stamp = None
            # rospy.loginfo("odom: Received Odometry")
        except Exception as e:
            rospy.logerr("odom: Error processing Odometry: %s", e)

    def publish_confidence_threshold(self, event):
        confidence_threshold_msg = Float64()
        confidence_threshold_msg.data = 0.4
        self.confidence_threshold_pub_.publish(confidence_threshold_msg)

    def run(self):
        rospy.loginfo("RealWorldNode running. Waiting for sensor messages...")
        rospy.spin()


@hydra.main(version_base=None, config_path="config", config_name="real_world_test")
def main(cfg: DictConfig):
    node = RealWorldNode(cfg)
    node.run()


if __name__ == "__main__":
    main()
