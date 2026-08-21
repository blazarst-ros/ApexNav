# ApexNav PX4 单一高度闭环设计

> 2026-08-18 22:39 动态复测修订：原先选择气压计高度参考的结论已被ULog证伪。
> 当前SITL使用 `EKF2_HGT_REF=3`、`EKF2_EV_CTRL=2`，由Gazebo launch-relative Z作为
> 外部视觉垂直测量；PX4 MPC仍是唯一控制器。以本修订和重大问题记录第21节为准。

## 问题与证据

最新运行中，ApexNav 记录的 MAVROS 相对高度约为 1.0 m，但 Gazebo 模型真值持续
下降到约 0.45 m。PX4 ULog 的 `vehicle_local_position_groundtruth` 与 Gazebo 一致，而 EKF
本地位置与真值的高度误差增大到约 0.60 m。原始气压计与 GPS 高度反而与真值
高度变化高度相关，说明主要故障在 PX4 SITL EKF 高度融合配置，不在 Gazebo
物理模型或 CSV 话题选择。

同时，supervisor 在同一个 `PositionTarget` 中发送固定 `position.z` 和基于同一
MAVROS Z 误差的 PD `velocity.z`。PX4 内部位置环已经会把 Z 位置误差转成速度
目标，因此外部 PD 不是前馈，而是第二个重复闭环。ULog 中已观测到两者叠加后
的过大垂直速度目标。

## 控制架构

1. `px4_native` 模式下，supervisor 只发送不变的世界坐标 `position.z = cruise_z`，
   并置位 `IGNORE_VZ`。PX4 本地位置控制器是唯一 Z 闭环。
2. 保留现有 XY 位置与速度前馈、yaw/yaw-rate 输入；二维规划器的 Z/VZ 永不
   透传到 PX4。
3. 删除 supervisor 外部 Z PD 参数，避免日后误开重复闭环。

## PX4 SITL 高度配置

`gazebo-classic_iris_depth_camera` 专用 airframe 使用：

- `EKF2_HGT_REF=3`、`EKF2_EV_CTRL=2`：只融合外部视觉垂直位置作为高度参考；
- `EKF2_BARO_NOISE=3.5`：气压计保留为次级高度源，不再让低噪声偏置模型主导；
- `MPC_THR_HOVER=0.70`：与此模型稳定悬停推力约 0.71 匹配，避免 hover-thrust
  estimator 从 0.50 起步并在旧上界 0.70 饱和。

supervisor 在进入 PRESTREAM 前通过 MAVROS `ParamGet` 验证完整高度/失联profile。配置不符时
拒绝解锁，明确告知需要重启 PX4，不让错误配置进入飞行。
验证结果仅对当前任务和 FCU 连接 epoch 有效；新任务或 FCU 断开/重连都必须重新
读取全部参数。

## Gazebo 真值一致性门控

仅在 Gazebo 专用配置中启用 `/gazebo/model_states` 诊断门控：分别以起飞前的 MAVROS
Z 和 Gazebo Z 为零点，比较两者 AGL。在已离地后，误差连续 1.0 s 超过
0.15 m 则阻止导航启动；任务执行期出现同样问题则停止规划器并进入非自动
降落的故障保持。Gazebo真值作为PX4 EKF的测量输入和独立一致性验证，不直接生成推力或
速度命令，因此仍保持单一PX4位置控制闭环。
超差 debounce 期间也不得从起飞进入 HOLD_READY；新任务必须用当前 disarmed 样本
重建真值零点，真值 stale、零点改变或时钟回跳都会重置连续超差计时。

## 验收标准

- 所有飞行 setpoint 均激活 PZ 且忽略 VZ。
- 解锁前能发现 PX4 高度 profile 不符，不进入 PRESTREAM/ARM。
- 起飞与巡航期间 MAVROS AGL 与 Gazebo AGL 绝对差稳态不超过 0.10 m，不允许
  持续超过 0.15 m。
- 通过高度一致性验收后，再判定 footprint collision 和局部路径拒绝是否仍
  属于规划器问题。
