#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import signal
import sys
import gzip
import json
import numpy as np
import rospy
from copy import deepcopy

# Habitat imports
import habitat
import habitat_sim
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.config.default import patch_config
import hydra
from omegaconf import DictConfig, OmegaConf
import habitat_sim.utils.common as utils

from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations.utils import observations_to_image

# ROS msgs
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, String

# Local imports
from habitat2ros import habitat_publisher 
from vlm.Labels import MP3D_ID_TO_NAME  # 确保你的环境中有此模块

def signal_handler(sig, frame):
    print("Shutting down Habitat Bridge...")
    rospy.signal_shutdown("Ctrl+C")
    sys.exit(0)

def transform_rgb_bgr(image):
    """将Habitat的RGB图像转换为OpenCV使用的BGR格式"""
    return image[:, :, [2, 1, 0]]

class RobotBridge:
    def __init__(self, robot_name, agent_index):
        """
        多机器人桥接类：封装单个机器人的ROS通信与物理控制
        :param robot_name: ROS命名空间名称 (e.g. 'robot_1')
        :param agent_index: Habitat中的agent索引 (e.g. 0)
        """
        self.robot_name = robot_name
        self.agent_index = agent_index
        self.ns_prefix = f"/{robot_name}"

        # 1. 初始化 ROS 发布器,自定义发布器(创建发布器对象而不是调用方法)(----module 5)
        self.pub = habitat_publisher.ROSPublisher(robot_name)
        
        # 订阅控制指令
        self.sub_vel = rospy.Subscriber(
            f"{self.ns_prefix}/cmd_vel", 
            Twist, 
            self.cmd_callback, 
            queue_size=10
        )
        
        # 辅助话题发布 (对应原代码中的 confidence_threshold_pub 和 label_pub)
        self.confidence_pub = rospy.Publisher(
            f"{self.ns_prefix}/detector/confidence_threshold", 
            Float64, 
            queue_size=10
        )
        self.label_pub = rospy.Publisher(
            f"{self.ns_prefix}/detector/label", 
            String, 
            queue_size=1, 
            latch=True
        )

        # 2. 物理状态初始化 (对应原模块1全局变量)
        self.cmd_vel = 0.0
        self.cmd_omega = 0.0
        self.camera_pitch = 0.0
        self.fusion_score = 0.3 # 默认初始融合分数

        # Habitat 速度控制器配置 (对应原模块3)
        self.vel_control = habitat_sim.physics.VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.ang_vel_is_local = True

        rospy.loginfo(f"[{self.robot_name}] Bridge initialized for Agent {self.agent_index}")

    def cmd_callback(self, msg):
        """接收 ROS 速度指令"""
        self.cmd_vel = msg.linear.x
        self.cmd_omega = msg.angular.z

    def publish_target_label(self, label):
        """发布目标语义标签 (对应原模块5的 label 发布)"""
        try:
            self.label_pub.publish(String(data=label))
            rospy.loginfo(f"[{self.robot_name}] Published target label: {label}")
        except Exception as e:
            rospy.logerr(f"[{self.robot_name}] Failed to publish label: {e}")

    def update_agent_physics(self, sim, time_step):
        """
        更新物理状态 (对应原模块6中的智能体运动控制部分,包括获取计算碰撞过滤，新位姿应用等)
        """
        agent = sim.agents[self.agent_index]
        
        # 设置目标速度 (Habitat 坐标系: -Z 为前方)
        self.vel_control.linear_velocity = np.array([0.0, 0.0, -self.cmd_vel])
        self.vel_control.angular_velocity = np.array([0.0, self.cmd_omega, 0.0])
        
        # 积分状态
        agent_state = agent.state
        previous_rigid_state = habitat_sim.RigidState(
            utils.quat_to_magnum(agent_state.rotation), agent_state.position
        )
        target_rigid_state = self.vel_control.integrate_transform(
            time_step, previous_rigid_state
        )
        end_pos = sim.step_filter(
            previous_rigid_state.translation, target_rigid_state.translation
        )
        
        agent_state.position = end_pos
        agent_state.rotation = utils.quat_from_magnum(target_rigid_state.rotation)
        agent.set_state(agent_state)

    def process_observations(self, all_obs):
        """
        从全局观测字典中提取本机器人数据，并注入额外信息 (对应原模块6的数据处理部分)
        """
        my_obs = {}
        # 定义传感器后缀映射 (适配 agent_{i}_{sensor} 格式)
        sensor_mapping = {
            "rgb": ["rgb", "color_sensor", "rgb_sensor"],
            "depth": ["depth", "depth_sensor"],
        }
        prefix = f"agent_{self.agent_index}_"

        # 1. 提取传感器数据
        for obs_key, potential_suffixes in sensor_mapping.items():
            found = False
            for suffix in potential_suffixes:
                full_key = prefix + suffix
                if full_key in all_obs:
                    my_obs[obs_key] = all_obs[full_key]
                    found = True
                    break
                # 兼容单机配置 (无前缀情况)
                if not found and self.agent_index == 0:
                    if suffix in all_obs:
                        my_obs[obs_key] = all_obs[suffix]
                        found = True
                        break

        # 2. 提取位姿 (GPS/Compass)
        gps_key = f"{prefix}gps"
        compass_key = f"{prefix}compass"
        
        if gps_key in all_obs:
            my_obs["gps"] = all_obs[gps_key]
        elif "gps" in all_obs and self.agent_index == 0:
            my_obs["gps"] = all_obs["gps"]

        if compass_key in all_obs:
            my_obs["compass"] = all_obs[compass_key]
        elif "compass" in all_obs and self.agent_index == 0:
            my_obs["compass"] = all_obs["compass"]

        # 3. 数据注入与发布 (完全对齐原代码逻辑)
        if "rgb" in my_obs and "depth" in my_obs:
            # 格式转换
            my_obs["rgb"] = transform_rgb_bgr(my_obs["rgb"])
            
            # 注入原代码中手动添加的字段
            my_obs["camera_pitch"] = self.camera_pitch
            my_obs["linear_velocity"] = self.cmd_vel
            my_obs["angular_velocity"] = self.cmd_omega
            
            # 发布观测数据
            self.pub.habitat_publish_ros_topic(my_obs)
            
            # 发布置信度阈值 (原代码逻辑: 每次发布观测后发布一次阈值)
            conf_msg = Float64()
            conf_msg.data = 0.5 # 原代码固定为 0.5, 若需动态可改为 self.fusion_score
            self.confidence_pub.publish(conf_msg)


@hydra.main(version_base=None, config_path="config", config_name="habitat_vel_control")
def main(cfg: DictConfig) -> None:
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("habitat_bridge_multi", anonymous=True)

    # =========================== 模块 2: 加载数据集映射 (迁移自原代码) ===========================
    # 这一步对于 real_world_test_habitat.py 的正常工作至关重要
    try:
        with gzip.open("data/datasets/objectnav/mp3d/v1/val/val.json.gz", "rt", encoding="utf-8") as f:
            val_data = json.load(f)
        category_to_coco = val_data.get("category_to_mp3d_category_id", {})
        id_to_name = {
            category_to_coco[cat]: MP3D_ID_TO_NAME[idx]
            for idx, cat in enumerate(category_to_coco)
        }
        rospy.loginfo("Loaded MP3D category mappings.")
    except Exception as e:
        rospy.logwarn(f"Failed to load dataset mappings: {e}. Labels might be raw IDs.")
        category_to_coco = {}
        id_to_name = {}

    # =========================== 模块 3: 自动修正配置 (Config Patching) ===========================
    cfg = patch_config(cfg)
    
    # [新增] 自动将单机配置转换为双机配置，防止因 YAML 未修改而报错
    if "agents" in cfg.habitat.simulator and "main_agent" in cfg.habitat.simulator.agents:
        rospy.loginfo("Detected single-agent config. Auto-converting to multi-agent (agent_0, agent_1)...")
        with habitat.config.read_write(cfg):
            main_agent_config = cfg.habitat.simulator.agents.main_agent
            # 创建 agent_0
            cfg.habitat.simulator.agents.agent_0 = main_agent_config
            # 创建 agent_1
            cfg.habitat.simulator.agents.agent_1 = main_agent_config
            # 删除旧的 key
            del cfg.habitat.simulator.agents.main_agent
    
    # 原代码中的测量项添加 (保留)
    with habitat.config.read_write(cfg):
        cfg.habitat.task.measurements.update(
            {
                "top_down_map": TopDownMapMeasurementConfig(
                    map_padding=3,
                    map_resolution=256,
                    draw_source=True,
                    draw_border=True,
                    draw_shortest_path=True,
                    draw_view_points=True,
                    draw_goal_positions=True,
                    draw_goal_aabbs=False,
                    fog_of_war=FogOfWarConfig(draw=True, visibility_dist=5.0, fov=79),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )

    # =========================== 环境初始化 ===========================
    try:
        env = habitat.Env(cfg)
    except Exception as e:
        rospy.logerr(f"Habitat initialization failed: {e}")
        return

    sim = env.sim
    sim.set_gravity(np.array([0.0, 0.0, 0.0]))

    # 设置 Episode (对应原模块4)
    env_count = cfg.test_epi_num
    if env_count > 0:
        env.current_episode = next(env.episode_iterator)
    
    observations = env.reset()
    rospy.loginfo("Habitat Environment Reset.")

    # =========================== 实例化机器人桥接 ===========================
    # 根据 config 中的 agents 数量动态创建
    agent_list = list(cfg.habitat.simulator.agents.keys()) # ['agent_0', 'agent_1']
    agent_list.sort() # 确保 0 在前
    
    robots = []
    for idx, agent_key in enumerate(agent_list):
        # 假设 ROS 命名为 robot_1, robot_2...
        robot_name = f"robot_{idx+1}"
        robots.append(RobotBridge(robot_name, idx))

    # =========================== 标签处理与发布 ===========================
    # 获取当前 Episode 的目标类别
    raw_label = env.current_episode.object_category
    label = raw_label

    # 执行映射 (raw -> coco_id -> name)
    if label in category_to_coco:
        coco_id = category_to_coco[label]
        label = id_to_name.get(coco_id, label)
    
    # 为所有机器人发布相同的目标标签 (假设是协作任务)
    for robot in robots:
        robot.publish_target_label(label)

    # =========================== 主控制循环 ===========================
    fps = 30.0 # 对齐原代码 FPS
    time_step = 1.0 / fps
    rate = rospy.Rate(fps)

    rospy.loginfo("Starting Multi-Robot Loop...")

    while not rospy.is_shutdown() and not env.episode_over:
        # 1. 物理层更新
        for robot in robots:
            robot.update_agent_physics(sim, time_step)

        # 2. 渲染层步进
        # 构建多机动作字典，Key 必须与 config 中的 agents key 一致
        action_dict = {}
        for agent_key in agent_list:
            # 必须确保 config 中定义了 'move_forward'
            action_dict[agent_key] = "move_forward"
            
        all_observations = env.step(action_dict)

        # 3. 数据处理与发布
        for robot in robots:
            robot.process_observations(all_observations)

        rate.sleep()

    env.close()

if __name__ == "__main__":
    main()
    
"""修改说明：
1. 创建了 RobotBridge 类，封装了单个机器人的 ROS 通信和物理控制逻辑
2. 在 main 函数中根据 config 中的 agents 动态创建 RobotBridge 实例，支持任意数量的机器人（前提是 config 中正确配置了 agents）。
3. 将原模块5中的 label 发布逻辑移入 RobotBridge 类的 publish_target_label 方法中，确保每个机器人都能独立发布自己的标签。
4. 将原模块6中的智能体运动控制逻辑移入 RobotBridge 类的 update_agent_physics 方法中，确保每个机器人都能独立处理自己的物理状态更新。
5. 在主循环中，首先调用每个机器人的 update_agent_physics 方法更新物理状态，然后统一调用 env.step 获取新的观测数据，最后调用每个机器人的
process_observations 方法处理并发布观测数据。
6. 添加了对单机配置的自动转换逻辑，确保即使用户没有修改 YAML 文件，也能正确运行多机器人版本。
7. 保留了原代码中的数据处理和发布逻辑，确保功能完全对齐原代码，同时增强了代码的模块化和可扩展性。
"""
"""对照检查：
1.全局变量vs类属性：控制变量，ROS 发布器，观测缓存分别进行了封装，隔离，优化
2.物理运动控制独立出来
3.数据处理与发布独立出来
4.标签映射
"""
"""通信话题与外部接口改动说明:
速度指令	/cmd_vel	                  /{robot_name}/cmd_vel	
RGB 图像	/habitat/camera_rgb	         /{robot_name}/habitat/camera_rgb	需修改 habitat_publisher.py
深度图像	/habitat/camera_depth	     /{robot_name}/habitat/camera_depth	
里程计	/habitat/odom	                /{robot_name}/habitat/odom	
检测阈值/detector/confidence_threshold	/{robot_name}/detector/confidence_threshold	
目标标签/detector/label	                /{robot_name}/detector/label
"""