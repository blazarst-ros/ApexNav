# ApexNav 多机地图数据流核验

本目录保存运行时数据流证据。系统仅应启动一个 `exploration_node`：两个
`agent_X` 的数据通过该节点写入同一张 SDF/Object/ValueMap。不要分别启动多个
planner，否则会形成三张独立地图并产生重复发布者。

## 使用顺序

在已启动 ROS master、planner 与仿真后，按顺序执行：

```bash
cd /home/blazarst/ApexNav/RuntimeData
./capture_ros_data.sh recognition
./capture_ros_data.sh object_filter
./capture_ros_data.sh depth_mapping
./capture_ros_data.sh shared_map
./capture_ros_data.sh planning
./capture_ros_data.sh pitch
```

每次命令都会创建 `capture_<阶段>_<时间戳>/`；保持运行一段包含机器人运动与检测的
时间后按 `Ctrl-C` 停止。每个目录含有 `*.bag`、`topic_manifest.txt`、各话题的
`rostopic info` 和 10 秒 `rostopic hz` 统计。可用下列命令复查：

```bash
rosbag info RuntimeData/capture_<阶段>_<时间戳>/<阶段>.bag
rosbag play --clock RuntimeData/capture_<阶段>_<时间戳>/<阶段>.bag
```

对单一机器人先将下面的 `{i}` 替换成 `0` 或 `1`，例如：

```bash
rostopic hz /detector/agent_0/clouds_with_scores
rostopic echo -n 1 /object/cluster_status
rostopic echo -n 1 /ros/agent_0/exploration_strategy
```

## Pitch 门槛复现

`pitch` 阶段只保存验证对象写入所需的位姿、两个 MapROS 门槛角、检测云、对象过滤云和对象簇状态：

```bash
./capture_ros_data.sh pitch
```

MapROS 发布的两个角度话题分别是：

```text
/map_ros/agent_0/camera_pitch
/map_ros/agent_1/camera_pitch
```

它们是 C++ 对 `sensor_pose` 做 `eulerAngles(2, 1, 0)` 后计算的同一角度。对象云回调仅在该值
不小于 `1.5 rad` 时继续写入；小于该值时，检测云会被提前丢弃。运行时可直接观察：

```bash
rostopic echo /map_ros/agent_0/camera_pitch
rostopic echo /map_ros/agent_1/camera_pitch
```

## 0. 拓扑前置检查

```bash
rosnode list
rosnode info /exploration_node
rostopic list | sort | tee RuntimeData/topic_list_$(date +%Y%m%d_%H%M%S).txt
```

判定：仅一个 `/exploration_node`，且它订阅 `agent_0` 和 `agent_1` 的深度、位姿、检测云、
ITM 分数和里程计。`/detector/agent_X/clouds_with_scores` 必须各有一个发布者；仿真模式
不要同时运行 `real_world_perception.py`，否则会产生重复检测发布者。

## 1. 识别输入与识别结果

数据流：`/habitat/agent_X/camera_rgb` 与深度图进入 Python 识别；检测结果经
`/detector/agent_X/clouds_with_scores` 进入 MapROS；语义相关性经
`/clip/agent_X/cosine_score` 写入 ValueMap。`/stage1/detector/detection` 是 Python
的可读诊断记录，**不被 C++ MapROS 订阅**，不能用它单独证明地图已写入。

| 话题 | 类型 | 判断 |
|---|---|---|
| `/habitat/agent_X/camera_rgb` | `sensor_msgs/Image` | 图像持续更新，目标应可见。 |
| `/habitat/agent_X/camera_depth` | `sensor_msgs/Image` | 与位姿时间接近；MapROS 同步窗口为 `0.05 s`。 |
| `/detector/agent_X/clouds_with_scores` | `plan_env/MultipleMasksWithConfidence` | `point_clouds`、`confidence_scores`、`label_indices` 长度相等；`label_indices=0` 为目标类。 |
| `/clip/agent_X/cosine_score` | `std_msgs/Float64` | 有有限数值；仅收到该值后才更新 ValueMap。 |
| `/stage1/detector/detection` | `plan_env/Stage1Detection` | 检查目标、top1、分数、距离和 mask 面积，作为识别侧证据。 |

阶段通过条件：三个 agent 的检测云及深度位姿均有频率；检测消息数组长度一致；
`class_names[0]` 是当前目标。若只有 Stage1 话题而没有 `clouds_with_scores`，识别可能正常但地图对象层没有输入。

## 2. 对象云过滤与对象地图写入

数据流：检测云先检查相机俯视姿态（pitch 必须不小于 `1.5 rad`），再经体素滤波
`0.04 x 0.04 x 0.06 m`、最大距离 `4.89 m`、欧氏聚类（容差 `0.15 m`、至少 `6` 点，
保留最大簇），最后写入共享 ObjectMap。

| 话题 | 含义 | 判断 |
|---|---|---|
| `/grid_map/all_object_cloud` | 进入 C++ 回调后的原始检测云汇总 | 有点说明检测云已到达 MapROS；它是最新一次 agent 回调，不是双机累计值。 |
| `/grid_map/filtered_object_cloud` | 距离、体素和聚类后的检测云 | 应有点且明显对应输入；为空表示被姿态、距离或聚类过滤。 |
| `/grid_map/over_depth_object_cloud` | 超量程目标点的短期一致性跟踪 | 有点时说明目标主要超出 `4.89 m`，不能正常写入对象簇。 |
| `/object/cluster_status` | 共享 ObjectMap 的最终簇状态 | `clusters` 非空才证明对象已经融合写入。 |
| `/grid_map/semantic_objects` | 按最终簇标签着色的语义云 | 应与 `cluster_status` 一致。 |
| `/grid_map/occupancy_object` | 对象占据栅格 | 有点说明对象簇已投影至地图。 |

阶段通过条件：`all_object_cloud` 有点后，`filtered_object_cloud` 仍有点，并随后能在
`cluster_status` 看到同一对象簇。若第一项有点而第二项没有，优先检查相机 pitch、距离与 DBSCAN；
若第二项有点而簇状态为空，检查对象融合阈值与 `object/min_observation_num=2`。

## 3. 深度过滤与占据地图写入

数据流：每个 agent 的 `camera_depth + sensor_pose` 以 `0.05 s` 同步，深度投影为世界系云，
经体素滤波 `0.04 x 0.04 x 0.10 m`、高度区间 `0.28 < z < 1.18 m`、半径离群滤波
（`0.3 m`、35 邻居）后写入共享占据栅格；再清理与膨胀局部地图并更新 ESDF。

| 话题 | 类型 | 判断 |
|---|---|---|
| `/habitat/agent_X/sensor_pose` | `nav_msgs/Odometry` | 必须和深度同步，坐标应落在地图边界内。 |
| `/habitat/agent_X/camera_depth` | `sensor_msgs/Image` | 连续有效深度；超范围深度会被截到 `4.99 m`。 |
| `/grid_map/depth_cloud` | `sensor_msgs/PointCloud2` | 深度投影后的世界系云。 |
| `/grid_map/filtered_depth_cloud` | `sensor_msgs/PointCloud2` | 高度和离群滤波后的障碍物点。 |
| `/grid_map/occupied` | `sensor_msgs/PointCloud2` | 共享地图中的障碍格。 |
| `/grid_map/free`、`/grid_map/unknown` | `sensor_msgs/PointCloud2` | 共享地图中的自由和未知格。 |
| `/grid_map/occupied_inflate`、`/grid_map/esdf` | `sensor_msgs/PointCloud2` | 规划实际避障使用的膨胀障碍和距离场。 |

阶段通过条件：每个 agent 的输入都在更新，且 `filtered_depth_cloud` 与环境障碍相符，
`occupied/free/unknown` 随机器人移动变化。若原始深度云有点而过滤云为空，检查高度范围、
传感器位姿、深度缩放和离群阈值；若过滤云有点而占据图不变，检查地图边界及 SDF 写入日志。

## 4. 双机共享对象/语义地图

数据流：三个 agent 的对象回调和深度回调都在 `map_mutex_` 保护下更新同一张 ObjectMap、
SDFMap 与 ValueMap；可视化每 `0.25 s` 发布共享快照。对象观察融合还使用每台机器人当前的
ITM 分数。

| 话题 | 判断 |
|---|---|
| `/grid_map/value_map` | 非零 intensity 仅在可通行自由格显示，证明 ITM 语义值已经写入。 |
| `/object/cluster_status` | 逐项检查 `cluster_id`、`centroid`、`state`、`best_label_name` 和 labels。 |
| `/object/cluster_markers` | RViz 中核对每个簇的位置、状态和双机观测后的合并情况。 |
| `/grid_map/semantic_objects` | 与 cluster 状态的目标/干扰/不确定类别对应。 |
| `/grid_map/all_object_cloud` | 仅表示最后一个回调，不能据此判断双机融合总量；融合结论以 cluster_status 为准。 |

阶段通过条件：从不同 `agent_X` 观测同一物体后，`cluster_status` 应聚合为同一或空间合理的簇，
而非按 agent 形成三份地图。每次 episode reset 后各可视化话题应发布空云，不能遗留上一回合数据。

## 5. 规划输入、决策和动作输出

规划读取共享 SDF/Object/ValueMap、每个 agent 的 odom，并为三台机器人按序规划。当前并无
“frontier 列表”ROS 话题，因此用地图快照、策略 JSON 和动作闭环共同确认规划。

| 话题 | 类型 | 判断 |
|---|---|---|
| `/habitat/agent_X/odom` | `nav_msgs/Odometry` | 规划起点持续更新。 |
| `/ros/agent_X/exploration_strategy` | `std_msgs/String` | JSON 含 `mode`、`target_id`、`semantic_score`、`path_length`、`target_pos`；有合理目标和路径长度。 |
| `/habitat/plan_action_agent_X` | `std_msgs/Int32` | 每台机器人都有动作输出，且不是持续 STOP。 |
| `/habitat/state`、`/ros/state_all` | `std_msgs/Int32` / `Int32MultiArray` | 仿真与双机 FSM 状态闭环一致。 |
| `/ros/agent_X/expl_result`、`/ros/expl_result_all` | `std_msgs/Int32` / `Int32MultiArray` | 确认每台机器人的探索结果。 |
| `/robot_agent_X` | `visualization_msgs/Marker` | RViz 核对机器人位置是否与地图同一 world 坐标系。 |

阶段通过条件：地图中存在可达自由区域和有效 ESDF 时，三个 `exploration_strategy` 均持续给出合理路径，
三个 action 话题都有动作且 odom 随之变化。若策略有目标而无动作，检查 FSM state；若动作存在但 odom
不变，问题在仿真动作执行链路；若策略路径长度异常大或无目标，回到深度/对象地图阶段检查可达性。

## 关键源码位置

* `src/planner/plan_env/src/map_ros.cpp`：双机订阅、所有过滤门槛、共享地图写入与可视化发布。
* `src/planner/plan_env/src/object_map2d.cpp`：对象簇融合和置信度更新。
* `src/planner/exploration_manager/src/exploration_fsm.cpp`：双机顺序规划、策略 JSON 与动作发布。
* `habitat_evaluation.py`：仿真端对每个 agent 发布检测云与 ITM 分数。
