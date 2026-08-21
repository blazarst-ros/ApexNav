# ApexNav 稳定巡航实施计划

1. 为扫描高度参考、原生 PX4 reference、渐进轨迹进度、制动稳定门控、目标锁定和轨迹
   连续性补失败回归测试。
2. 在 trajectory server 增加 PX4-native reference 与闭环进度调速，保留 legacy MPC，且在
   新轨迹和 stop 时清除控制器历史。
3. 在 supervisor 统一强制 `cruise_z`，收紧扫描入口与恢复带，并加入 yaw 加速度限制和
   掉高时地图融合门控。
4. 在 exploration FSM 接入执行进度、制动稳定门控、起点连续性验证和 10 s 目标迟滞。
5. 用中心膨胀与完整定向 footprint 的组合检查保护执行轨迹，再把 Gazebo 全局膨胀微调到
   0.45 m。
6. 编译全工作区、运行 Python/nosetest 和 C++ gtest，检查 launch 展开参数和 diff。
7. 将运行结论、根因、实现、验证及下一轮验收工作流追加到重大问题记录。
