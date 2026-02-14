"""结合了 Habitat 仿真环境 与 ROS (Robot Operating System) 的机器人速度控制程序，
核心功能是在 Habitat 仿真环境中驱动智能体移动，
并将环境观测数据发布到 ROS 话题，
同时接收 ROS 的速度控制指令来操控智能体。
是同模拟器交互的核心
"""
import os
import signal
import gzip
import json
import time

import habitat 
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from omegaconf import DictConfig
from habitat.config.default import patch_config
import hydra  # noqa
from habitat2ros import habitat_publisher
import rospy
from copy import deepcopy
from std_msgs.msg import Float64, String
from vlm.Labels import MP3D_ID_TO_NAME
from geometry_msgs.msg import Twist
import habitat_sim
from habitat_sim.utils import common as utils # type: ignore

from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
)
from habitat.utils.visualizations.utils import observations_to_image


def signal_handler(sig, frame):#中断信号处理
    print("Ctrl+C detected! Shutting down...")
    rospy.signal_shutdown("Manual shutdown")
    os._exit(0)


def transform_rgb_bgr(image):#图像格式转换Habitat 输出的 RGB 图像通道顺序为 R-G-B，转换为 B-G-R（适配 OpenCV/ROS 的图像格式）
    return image[:, :, [2, 1, 0]]


def publish_observations(event):#ROS 数据发布（大概率由时间信号触发）
    global msg_observations, fusion_score
    global ros_pub, confidence_threshold_pub
    tmp = deepcopy(msg_observations)  #深度拷贝，保证发布的是 "发布瞬间的完整数据"
    ros_pub.habitat_publish_ros_topic(tmp)#自定义的发布方法
    msg = Float64()
    msg.data = fusion_score
    confidence_threshold_pub.publish(msg)


def cmd_vel_callback(msg):#ROS 速度指令订阅
    global cmd_vel, cmd_omega
    cmd_vel = msg.linear.x
    cmd_omega = msg.angular.z


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="habitat_vel_control",
)
def main(cfg: DictConfig) -> None:   #DictConfig 不是 YAML 配置项：是 Hydra 解析 YAML 后生成的 Python 增强字典对象
    
    # ===========================模块 1：全局变量定义与初始化=================
    global msg_observations, fusion_score
    global ros_pub, confidence_threshold_pub
    global obj_point_cloud
    global obj_point_cloud_pub
    global cmd_vel, cmd_omega

    cmd_vel = 0.0
    cmd_omega = 0.0

    #===========================模块 2：加载数据集&类别映射=================
    with gzip.open(
        "data/datasets/objectnav/mp3d/v1/val/val.json.gz", "rt", encoding="utf-8"
    ) as f:
        val_data = json.load(f)
    category_to_coco = val_data.get("category_to_mp3d_category_id", {})
    #字典，键是Habitat/MP3D 数据集的类别名（比如cabinet、chair），值是对应的 COCO 数据集类别 ID
    id_to_name = {
        category_to_coco[cat]: MP3D_ID_TO_NAME[idx]
        for idx, cat in enumerate(category_to_coco)
    }



    #===========================模块 3：Habitat 仿真环境初始化=================
    cfg = patch_config(cfg)
    env_count = cfg.test_epi_num
    print(env_count)
    cfg_rgb_sensor = cfg.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor

    height = cfg_rgb_sensor["height"]
    width = cfg_rgb_sensor["width"]
    fusion_score = 0.3 # 初始化融合评分（后续用于ROS发布到/detector/confidence_threshold）

    # Control-related parameters
    fps = 30.0
    time_step = 1.0 / fps

    # No LLM output directory required when LLM is disabled/removed

    # Add top_down_map and collision (measurements添加顶视图地图和碰撞检测测量项)
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
                    fog_of_war=FogOfWarConfig(
                        draw=True,
                        visibility_dist=5.0,
                        fov=79,
                    ),
                ),
                "collisions": CollisionsMeasurementConfig(),
            }
        )
        
    
    env = habitat.Env(cfg)
    sim = env.sim
    sim.set_gravity(np.array([0.0, 0.0, 0.0]))
    vel_control = habitat_sim.physics.VelocityControl()
    vel_control.controlling_lin_vel = True
    vel_control.controlling_ang_vel = True
    vel_control.lin_vel_is_local = True
    vel_control.ang_vel_is_local = True

    print("Environment creation successful")
    
    #==========模块 4：仿真回合初始化与观测数据缓存(用于指定起始化位点episode=1,2,3...)==========
    while env_count:
        env.current_episode = next(env.episode_iterator)
        env_count -= 1
    observations = env.reset()
    observations["rgb"] = transform_rgb_bgr(observations["rgb"])

    agent = sim.agents[0] # 获取第一个智能体（主智能体）
    info = env.get_metrics()# 获取环境指标（如位置、碰撞状态）
    frame = observations_to_image(observations, info)# 观测数据转可视化图像


    camera_pitch = 0.0
    observations["camera_pitch"] = camera_pitch
    observations["linear_velocity"] = 0.0
    observations["angular_velocity"] = 0.0
    msg_observations = deepcopy(observations) # 深拷贝缓存，避免原数据被修改


    #===================模块5 ROS 通信链路搭建（发布与订阅）============================
    #自定义发布器(创建发布器对象而不是调用方法)
    ros_pub = habitat_publisher.ROSPublisher() 
    cmd_sub = rospy.Subscriber("/cmd_vel", Twist, cmd_vel_callback, queue_size=10)
    timer = rospy.Timer(rospy.Duration(0.1), publish_observations)# 记录循环开始时间（用于性能监控）
    itm_score_pub = rospy.Publisher("/blip2/cosine_score", Float64, queue_size=10)
    # clouds-with-scores publisher removed (not used in this script)
    confidence_threshold_pub = rospy.Publisher(
        "/detector/confidence_threshold", Float64, queue_size=10
    )
    # Publish the target label so other nodes can subscribe（发布目标标签如：cabinet）
    label_pub = rospy.Publisher("/detector/label", String, queue_size=1, latch=True)

    print("Agent stepping around inside environment.")
    label = env.current_episode.object_category

    if label in category_to_coco:
        coco_id = category_to_coco[label]
        label = id_to_name.get(coco_id, label)

    # Publish the selected label so external nodes (e.g. real-world node) can receive it
    
    try:
        label_pub.publish(String(data=label))
        rospy.loginfo("Published target label: %s", label)
    except Exception as e:
        print(f"Failed to publish label: {e}")
        
        
        
    #========================模块6 智能体运动控制主循环========================
    rate = rospy.Rate(fps)

    tmp_cnt = 0
    while not rospy.is_shutdown() and not env.episode_over:
        loop_begin_time = rospy.Time.now()
        object_mask = np.zeros((height, width), dtype=np.uint8)
        # 初始化速度控制器为0
        vel_control.linear_velocity = np.array([0.0, 0.0, 0.0])  # y+ None x-
        vel_control.angular_velocity = np.array([0.0, 0.0, 0.0])
        timer.shutdown()
        
        # 从ROS全局变量读取速度指令，赋值给物理控制器
        vel_control.linear_velocity = np.array([0.0, 0.0, -cmd_vel])#智能体前进方向是z 轴负方向
        vel_control.angular_velocity = np.array([0.0, cmd_omega, 0.0])#智能体绕 y 轴旋转
        
        # 硬编码逻辑：仅仅前4秒强制左转90度（测试用）
        tmp_cnt += 1
        if tmp_cnt >= 1 and tmp_cnt <= 4.0 * fps + 5:
            vel_control.angular_velocity = np.array([0.0, np.pi / 2.0, 0.0])


        #位姿更新流程：当前位姿 → 计算目标位姿 → 碰撞过滤 → 应用新位姿
        
        agent_state = agent.state# 1. 获取当前智能体状态（位姿：位置+旋转）
        previous_rigid_state = habitat_sim.RigidState(# 2. 封装为Habitat刚体状态（便于物理计算）
            utils.quat_to_magnum(agent_state.rotation), agent_state.position
        )
        target_rigid_state = vel_control.integrate_transform(# 3. 积分计算目标位姿（基于速度+时间步）
            time_step, previous_rigid_state
        )
        end_pos = sim.step_filter(# 4. 碰撞过滤（避免智能体穿墙/穿障碍物）
            previous_rigid_state.translation, target_rigid_state.translation
        )
        # 5. 更新智能体位姿
        agent_state.position = end_pos
        agent_state.rotation = utils.quat_from_magnum(target_rigid_state.rotation)
        agent.set_state(agent_state)

        rospy.loginfo_throttle(5.0, f"I'm finding {label}")

        observations = env.step(HabitatSimActions.move_forward)# 步进仿真环境：获取新的观测数据（RGB、深度、位姿等）

        habitat_env_time = rospy.Time.now() - loop_begin_time

        info = env.get_metrics()
        # 补充观测数据字段（供ROS发布）
        observations["camera_pitch"] = camera_pitch
        observations["linear_velocity"] = cmd_vel
        observations["angular_velocity"] = cmd_omega
        ros_pub.habitat_publish_ros_topic(observations)# 发布观测数据到ROS（自定义封装的发布器）
        msg = Float64()# 发布置信度阈值到ROS（固定0.5），实际应用中可以根据观测数据动态调整
        msg.data = 0.5
        confidence_threshold_pub.publish(msg)
        # 格式转换+清理临时字段（避免后续处理报错）
        observations["rgb"] = transform_rgb_bgr(observations["rgb"])
        del observations["camera_pitch"]
        del observations["linear_velocity"]
        del observations["angular_velocity"]
        frame = observations_to_image(observations, info)

        if habitat_env_time.to_sec() >= time_step:
            print(
                f"env step time: {habitat_env_time.to_sec()*1000.0:.1f}ms VS {time_step*1000.0:.1f}ms"
            )

        rate.sleep()#30Hz 循环频率控制

    env.close()


if __name__ == "__main__":  
    """只有直接运行该脚本时（main），
    才执行后续的信号注册、ROS 初始化、调用main()等逻辑；
    如果该脚本被import到其他文件，
    这部分代码不会执行，
    保证代码的模块化"""
    
    signal.signal(signal.SIGINT, signal_handler)
    rospy.init_node("habitat_ros_publisher", anonymous=True)

    try:
        main()
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        rospy.signal_shutdown("Shutdown due to error")
        os._exit(1)
