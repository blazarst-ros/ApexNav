# Lite ApexNav Gazebo/真机鲁棒全链路方案

## 目标与边界

本链路将现有二维ApexNav规划用于PX4 Iris定高飞行：规划器只处理 `x/y/yaw`，PX4监督器保持 `z=1.0m`。
感知只使用YOLOE实例分割和CLIPITM图文匹配，不启用旧模型或旧模型回退。MAVROS EKF的ENU位姿是控制和建图基准；Gazebo truth只用于评测。

## 信息流

```text
Gazebo RGB/Depth + CameraInfo    MAVROS local odom
              \                   /
       HM3D虚拟相机重投影 + 位姿时间插值
                       |
      RGB/Depth/CameraInfo/CameraPose（同一源时间）
                       |
            YOLOE + CLIPITM异步推理
                       |
             SemanticObservation
                       |
       Occupancy / Object / Value Map
                       |
          FSM -> KinoAstar -> MPC
                       |
          /apexnav/planner/cmd_vel_raw
                       |
             PX4 Offboard监督器
                       |
              起飞/AUTO/HOLD/LAND
```

占据地图以深度帧频率独立更新，不等待VLM。语义消息携带源图像时间戳，返回后只允许更新匹配的历史地图帧。

## 相机模型

HM3D-v2继承配置为 `640x480、HFOV=79°`。按照仓库现有投影公式，Gazebo统一K矩阵为：

```text
[388.1910413097385, 0,                 320]
[0,                 422.0475153598262, 240]
[0,                 0,                   1]
```

Gazebo适配器从源CameraInfo生成OpenCV重投影表，RGB采用双线性插值，深度采用最近邻插值并保留无效0值。输出CameraInfo是唯一内参源。
真机不使用上述K矩阵，始终采用实际硬件标定；若需要统一虚拟相机，必须显式标定和重投影，不能只替换K。

## 时间和数据防御

- Gazebo RGB/Depth最大差5ms；真机最大差30ms。
- 位姿历史保留5秒，在传感器时刻对平移线性插值、对姿态SLERP；样本间隔不得超过100ms，向未来外推不得超过20ms。
- 正式链路禁止用 `Time(0)`、`now()`或接收时刻替代源时间。
- 零、重复、倒退、来自未来或超龄的数据直接丢弃并计入diagnostics。
- MapROS深度/位姿最大差10ms；VLM与历史地图帧最大差50ms。
- Gazebo必须启用 `/use_sim_time` 且 `/clock`有效，禁止墙钟和仿真时间混用。
- RGB、Depth和CameraInfo尺寸必须一致；任务中途改变尺寸或标定时停止更新并要求重新启动任务。
- 深度内部统一为CV_32FC1米。支持32FC1米、16UC1毫米和Habitat normalized三种显式模式；未知编码、NaN、Inf、0、负值和越界值不会进入地图。
- 点云缓存按实际分辨率动态分配，不再假设640x480。
- 语义工作队列长度为1，积压时只保留最新帧。目标标签变化会使旧generation结果失效。
- `SemanticObservation`原子携带YOLOE点云、置信度、label、CLIP分数、模型有效位和延迟；MapROS拒绝重复、乱序、过期和不同世界坐标系的消息。

## 坐标约定

- 世界系：`map`，MAVROS ENU。
- 机体系：`base_link`。
- 相机：`iris_camera_optical_frame`，满足ROS optical的 `z前/x右/y下`。
- 相机语义朝向使用光学+Z轴在世界XY平面的投影，不使用optical quaternion的Euler-Z。
- 默认Iris外参是平移 `[0.10, 0, 0.035]`、RPY `[-pi/2, 0, -pi/2]`；首次运行必须用点云方向和已知墙面复核。

## VLM与决策

模型服务提供 `/healthz`，客户端采用有限超时和有限重试，失败会返回上层，不会调用 `exit()`。RViz/diagnostics显示模型名、健康状态、目标、结果年龄、推理延迟和丢弃计数。

ObjectMap中目标置信度阈值为0.30且至少需要两次观测。CLIPITM只更新ValueMap和frontier价值，不能单独宣布找到目标。FSM明确发布探索、目标搜索、无frontier和到达目标结果；到达目标发布 `REACH_OBJECT=4` 并立即清除轨迹。原点附近障碍物不再被绕过。

Gazebo Iris限制为：水平速度0.25m/s、加速度0.30m/s²、yaw rate 0.40rad/s、平面footprint 0.70m×0.70m、地图膨胀0.08m、安全距离0.10m。建图障碍高度切片为0.35至1.65m。

## PX4生命周期和可靠停止

```text
WAIT_FCU -> PRESTREAM(>=2s) -> ARM -> OFFBOARD_TAKEOFF
         -> HOLD_READY -> AUTO -> HOLD/FAULT -> LAND -> DISARMED
```

监督器以30Hz持续发布MAVROS PositionTarget。AUTO将规划器前向速度按当前yaw转换为ENU `vx/vy`，保持1.0m高度并使用yaw_rate；起飞和HOLD锁定XY/yaw。

- cmd超过0.30s、odom超过0.20s、RGB/Depth超过0.50s：立即清除轨迹并HOLD。
- VLM单次推理失败只丢弃该语义结果；健康检查连续失效才进入FAULT。
- 正常完成HOLD 2秒后降落；关键故障HOLD 3秒后降落。
- 监督器切换 `AUTO.LAND`，确认landed后才disarm。
- PX4 offboard-loss failsafe应另行设为Land，形成独立保护层。

## 启动与验收

按README启动Gazebo/PX4、两个VLM和 `gazebo_full_stack.launch`。任务入口是 `/apexnav/mission/start`，首个验收目标固定为 `chair`。

必须完成以下测试：

1. 验证统一输出640x480且CameraInfo等于HM3D K矩阵。
2. 对已知墙面验证外参、optical轴、深度尺度和地图重合。
3. 延迟VLM时占据地图仍持续更新，迟到语义只作用于其源历史帧。
4. chair至少两次有效观测后才从frontier切换到目标搜索。
5. 分别停止clock、RGB、Depth、odom、YOLOE、CLIPITM和规划命令，检查diagnostics和HOLD/LAND时限。
6. 全流程完成预发送、解锁、1m起飞、探索、`REACH_OBJECT`、HOLD、降落和disarm。

任何测试中均不得出现旧CLIP分数用于新帧、最新TF冒充历史TF、停止后旧轨迹再次输出非零命令，或旧模型参与决策。
