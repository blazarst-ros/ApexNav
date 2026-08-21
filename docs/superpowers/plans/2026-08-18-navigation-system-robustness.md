# 飞行导航系统鲁棒性实施计划

Spec: `docs/superpowers/specs/2026-08-18-navigation-system-robustness-design.md`

## Global Constraints

- PX4 是唯一 Z 反馈控制器；所有飞行 setpoint 固定 PZ 且 `IGNORE_VZ`，Gazebo 真值
  只用于 SITL 审计和门控。
- `/mavros/setpoint_raw/local` 只能由 supervisor 发布；任何上游异常数据必须在 FCU
  边界再次校验。
- 传感器、地图和轨迹的源时间必须非零、不过期、不在未来且不回退；接收时间不能把
  陈旧源数据伪装成新鲜数据。
- 平滑后的最终轨迹必须在发布前通过完整中心占据和有向 footprint 检查；运行期检查
  是第二道防线，不应第一次发现静态地图上的既有碰撞。
- 操作员停止、故障、轨迹流中断均先清除旧轨迹并保持 XY/巡航 Z；是否受控降落由
  明确的故障策略决定。
- 每个 bugfix 遵循 RED→GREEN；不删除或覆盖工作区中已有的用户修改。

## Task 1: 加固 supervisor 的 FCU 最终边界

Files:

- `src/simulation/apexnav_gazebo/scripts/px4_mission_supervisor.py`
- `src/simulation/apexnav_gazebo/test/test_px4_mission_supervisor.py`

Requirements:

1. 先写失败测试证明：零/陈旧/未来/非有限 `trajectory_reference` 不能刷新命令 epoch，
   不能进入 `_setpoint(auto=True)`；合法 reference 仍被接受。
2. `_trajectory_reference_cb` 必须以 header source stamp 判定当前执行 epoch，拒绝延迟旧
   轨迹；reference 的 x/y/vx/vy/yaw/yaw_rate 全部必须有限。
3. `_setpoint(auto=True)` 在发布前再次做 finite 检查；异常时发布安全 hold，而不是
   NaN/Inf。不得破坏固定 PZ/IGNORE_VZ 契约。
4. 先写失败测试证明 `/clock` 回退后 `_call_limited` 能立即重新尝试服务，再修复负时间差
   被永久节流的问题。
5. PLAN_TRAJ 进入时清空旧 reference epoch。首 reference 等待逻辑使用 source stamp 和
   execution epoch；保留有界等待，不把 timeout 简单无限放宽。

## Task 2: 加固轨迹服务器消息入口

Files:

- `src/planner/trajectory_manager/include/trajectory_manager/trajectory_message_validation.h`
- `src/planner/trajectory_manager/src/traj_server.cpp`
- `src/planner/trajectory_manager/test/test_trajectory_message_validation.cpp`
- `src/planner/trajectory_manager/CMakeLists.txt`

Requirements:

1. 先写纯函数 gtest，覆盖空轨迹、order 非 7、三轴长度不一致、duration 非正/非有限、
   任一系数 NaN/Inf，以及完整合法消息。
2. 入口在任何数组索引或构造 `Trajectory` 前完成 shape/finite/positive validation；拒绝时
   不替换当前有效轨迹、不发布参考，并记录明确错误。
3. 验证 `start_time` 非零且不过度陈旧；允许未来 start_time 的正常 PREPARE 窗口。

## Task 3: 封闭规划旁路并增加最终轨迹安全验证

Files:

- `src/planner/exploration_manager/include/exploration_manager/robust_navigation_policy.h`
- `src/planner/exploration_manager/include/exploration_manager/exploration_fsm_traj.h`
- `src/planner/exploration_manager/src/exploration_fsm_traj.cpp`
- `src/planner/exploration_manager/test/test_robust_navigation_policy.cpp`

Requirements:

1. 删除 `/initialpose` 直接规划/发布控制入口；RViz Initial Pose 不能在 WAIT/AUTO/FINISH
   绕过 supervisor 和 FSM。用静态/纯函数测试证明订阅和直接发布旁路已不存在。
2. 在 `enforceTrajectoryLimits` 之后、保存/发布轨迹之前，按不大于 0.05 m 的弧长空间
   分辨率（同时设合理最大时间步）遍历完整 spline；每个样本检查中心 inflate occupancy
   和 `isCollisionPosYaw`（其将 UNKNOWN 视为碰撞）。失败返回 FAILED 并保持 PLAN_TRAJ。
3. 轨迹 yaw 使用速度方向；低速段保持上一有效 yaw，并验证终点，不能以当前 odom yaw
   掩盖曲线末端 footprint。
4. 将 NO_FRONTIER 协议统一：对“目标搜索耗尽且未找到”明确发布失败/不可达终态；planner
   和 supervisor 不得一方称 MISSION_COMPLETE、另一方称 FAULT。保留 REACH_OBJECT=4
   的成功 HOLD 路径。

## Task 4: 修复地图时间、self-filter 和膨胀一致性

Files:

- `src/planner/plan_env/include/plan_env/map_filter_policy.h`
- `src/planner/plan_env/include/plan_env/map_ros.h`
- `src/planner/plan_env/include/plan_env/sdf_map2d.h`
- `src/planner/plan_env/src/map_ros.cpp`
- `src/planner/plan_env/src/sdf_map2d.cpp`
- `src/planner/plan_env/test/test_map_filter_policy.cpp`
- `src/planner/plan_env/CMakeLists.txt`
- Gazebo/exploration launch files defining filter dimensions

Requirements:

1. 先写测试证明圆形 0.65 m self-filter 会删除 footprint 外真实点；改成与机体 yaw 对齐的
   0.70×0.70 m 有向 footprint（仅小数值容差），过滤与历史清理共用同一判定。
2. 地图点云在 depth callback 中使用 depth source stamp；周期地图快照使用最后一帧成功
   建图的 source stamp。第一次源帧之前允许空快照为零，但不能刷新健康状态。
3. supervisor 的 filtered map 健康只接受非零、非未来且年龄合格的 header stamp。
4. 重算受影响 inflation 区域时同时考虑相邻仍 occupied 的源，避免清除一个障碍时在另一
   障碍 halo 打洞；写两个重叠 halo 的回归测试。
5. 修正 `free_grids` 在 raycast 填充之前膨胀的 no-op 顺序。

## Task 5: 全链路契约、构建、复核与交付记录

Requirements:

1. 增加 launch/config 静态契约，核对唯一 MAVROS setpoint 发布者、world frame、Gazebo
   truth guard、PZ/VZ mask、轨迹/地图 source stamp 和 planner/supervisor result 枚举。
2. 运行 Python 单测/语法检查、全部 C++ gtest、`catkin_make` 与 `catkin_make run_tests`，
   使用 `catkin_test_results --all` 确认零失败。
3. 独立审查本轮 diff；修复所有 Critical/Important。
4. 将最新旧日志证据、根因、修改、静态验证、正确启动工作流和必须执行的新 SITL 动态
   验收追加到 `RuntimeData/APEXNAV_MAJOR_ISSUES_20260818.md`。不得把旧 capture 宣称为
   改后飞行成功。
