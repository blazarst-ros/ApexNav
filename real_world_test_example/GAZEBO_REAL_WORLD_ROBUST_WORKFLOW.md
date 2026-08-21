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
       FSM -> KinoAstar -> GCopter/MINCO
                       |
       source-stamped XY/yaw trajectory reference
                       |
             PX4 Offboard监督器
                       |
              起飞/AUTO/HOLD/LAND
```

占据地图以深度帧频率独立更新，不等待VLM。语义消息携带源图像时间戳，返回后只允许更新匹配的历史地图帧。只有 occupancy、自机清理和 inflation 全部完成后才发布 `/grid_map/commit`；监督器不再用预过滤点云冒充“地图已提交”。

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
- 位姿历史保留5秒，在传感器时刻对平移线性插值、对姿态SLERP；样本间隔不得超过100ms。为容纳MAVROS相对Gazebo相机的一至两个周期延迟，只允许使用最新位姿最多80ms，发布结果仍严格保留相机源时间。
- 正式链路禁止用 `Time(0)`、`now()`或接收时刻替代源时间。
- 零、重复、倒退、来自未来或超龄的数据直接丢弃并计入diagnostics。
- MapROS深度/位姿最大差10ms；VLM与历史地图帧最大差50ms。
- Gazebo必须启用 `/use_sim_time` 且 `/clock`有效，禁止墙钟和仿真时间混用。
- RGB、Depth和CameraInfo尺寸必须一致；任务中途改变尺寸或标定时停止更新并要求重新启动任务。
- 深度内部统一为CV_32FC1米。支持32FC1米、16UC1毫米和Habitat normalized三种显式模式；未知编码、NaN、Inf、0、负值和越界值不会进入地图。
- 点云缓存按实际分辨率动态分配，不再假设640x480。
- 语义工作队列长度为1，积压时只保留最新帧。目标标签变化会使旧generation结果失效。
- `SemanticObservation`原子携带YOLOE点云、置信度、label、CLIP分数、模型有效位和延迟；MapROS逐帧验证数组、label `[0,4]`、有限置信度、嵌套点云布局/坐标系/源时间和全部XYZ，任一异常整帧拒绝。
- 新任务或 `/clock` 回退会清空旧 occupancy、inflation、ESDF、ObjectMap、ValueMap 和历史；旧地图不能在新时间纪元复活。ESDF尚未完成时不发布混合的新占据/旧距离快照。

## 坐标约定

- 世界系：`map`，MAVROS ENU。
- 机体系：`base_link`。
- 相机：`iris_camera_optical_frame`，满足ROS optical的 `z前/x右/y下`。
- 相机语义朝向使用光学+Z轴在世界XY平面的投影，不使用optical quaternion的Euler-Z。
- 默认Iris外参是平移 `[0.10, 0, 0.035]`、RPY `[-pi/2, 0, -pi/2]`；首次运行必须用点云方向和已知墙面复核。

## VLM与决策

模型服务提供 `/healthz`，客户端采用有限超时和有限重试，失败会返回上层，不会调用 `exit()`。RViz/diagnostics显示模型名、健康状态、目标、结果年龄、推理延迟和丢弃计数。

ObjectMap中目标置信度阈值为0.30且至少需要两次观测。CLIPITM只更新ValueMap和frontier价值，不能单独宣布找到目标。FSM明确发布探索、目标搜索、无frontier和到达目标结果；到达目标发布 `REACH_OBJECT=4` 并立即清除轨迹。原点附近障碍物不再被绕过。

Gazebo Iris限制为：水平速度0.30m/s、加速度0.30m/s²、yaw rate 0.55rad/s、平面footprint 0.70m×0.70m、地图膨胀0.38m、安全距离0.10m。每段最终样条先用 Bernstein 导数凸包得到全区间速度/yaw-rate保守上界并完成时间缩放，再按不大于0.05m/0.05s的有证明步长检查UNKNOWN、膨胀栅格和完整有向footprint，避免固定采样漏掉点间尖峰。建图采用相机相对高度切片（下方0.18m至上方0.25m），避免假设PX4局部坐标中的地面恒为z=0；室内有效深度与射线长度限制为4.0m。

## PX4生命周期和可靠停止

```text
WAIT_FCU -> PRESTREAM(>=2s) -> ARM -> OFFBOARD_TAKEOFF
         -> 定高稳定确认 -> HOLD_READY -> AUTO -> HOLD/FAULT -> LAND -> DISARMED
```

深度建图和YOLOE/CLIPITM使用独立门控。起飞前不更新任务地图、不执行目标推理，但
监督器会要求两个模型服务的 `/healthz` 均已就绪。飞机达到1米巡航高度、误差不超过
0.05米、垂直速度不超过0.025m/s并稳定2.5秒后，直接进入 `HOLD_READY`；历史上会导致
降高的起始360度旋转已完全删除。随后开启mapping与VLM门控，等待当前任务的新地图和
新语义结果，旧任务/旧时钟纪元的数据不能放行。
`HOLD_READY` 保持定点悬停，等待
首个有效地图帧、当前目标语义结果和规划器握手完成后再进入 `AUTO`。进入
`HOLD/FAULT/LAND/DISARMED` 时立即重新禁用。

巡航高度以任务开始时的局部高度为零点，状态中报告非负相对高度。目标接近路径优先
保持0.85米距离并通过完整机体footprint碰撞检查；若该点不安全，则沿路径后退到最近的
无碰撞停靠点，避免把物体占据栅格作为轨迹终点。

监督器是 `/mavros/setpoint_raw/local` 的唯一发布者并以30Hz输出。AUTO使用规划器的
source-stamped ENU XY位置与速度前馈，同时始终发送固定巡航PZ并屏蔽VZ；PX4原生位置控制器
是唯一的Z控制器。Gazebo的launch-relative真值高度通过 `/mavros/vision_pose/pose` 作为
EKF外部视觉垂直测量，因此参与闭环的是实际Z，而不是会吸收慢速下降的气压计偏置。

- 首条轨迹参考允许0.80s独立握手；稳定执行后source stamp超过0.30s、odom超过0.20s、RGB/Depth超过0.50s：立即清除轨迹并进入有界FAULT。
- 规划目标至少连续失败3次且首失败已持续2秒才会更换；unsafe/reached仍立即解除。重规划阶段/总预算为10/20秒，可覆盖KinoAstar的2秒搜索上限与一次换目标，同时仍保证失败有界。
- VLM单次推理失败只丢弃该语义结果；健康检查连续失效才进入FAULT。
- 正常完成HOLD 2秒后降落；关键故障HOLD 3秒后降落。正常数据链下以0.25m/s的固定XY/PZ轨迹缓降；输入失效或10秒超时才切换 `AUTO.LAND`，确认landed后才disarm。
- SITL airframe固定 `EKF2_HGT_REF=3(Vision)`、`EKF2_EV_CTRL=2(仅垂直位置)`、`EKF2_BARO_NOISE=3.5`、`MPC_THR_HOVER=0.70`、`MPC_LAND_SPEED=0.60`、`COM_OF_LOSS_T=1.0`、`COM_OBL_RC_ACT=4(Land)`、`COM_RC_IN_MODE=4`，监督器逐任务通过MAVROS参数服务核对；参数错误时绝不预发送或解锁。

## 启动与验收

每次动态验收都完整退出并重启Gazebo/PX4/MAVROS/planner，再按README启动两个VLM和 `gazebo_full_stack.launch`；不能用旧PX4参数或已处于FINISH的planner复测。任务入口是 `/apexnav/mission/start`，首个验收目标固定为 `chair`。

必须完成以下测试：

1. 验证统一输出640x480且CameraInfo等于HM3D K矩阵。
2. 对已知墙面验证外参、optical轴、深度尺度和地图重合。
3. 延迟VLM时占据地图仍持续更新，迟到语义只作用于其源历史帧。
4. chair至少两次有效观测后才从frontier切换到目标搜索。
5. 分别停止clock、RGB、Depth、odom、YOLOE、CLIPITM和规划命令，检查diagnostics和HOLD/LAND时限。
6. 全流程完成预发送、解锁、1m起飞、探索、`REACH_OBJECT`、HOLD、降落和disarm。

任何测试中均不得出现旧CLIP分数用于新帧、最新TF冒充历史TF、停止后旧轨迹再次输出非零命令，或旧模型参与决策。
