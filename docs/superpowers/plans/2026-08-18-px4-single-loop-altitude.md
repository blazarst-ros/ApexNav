# PX4 单一高度闭环实施计划

1. 在 supervisor Python 测试中先增加三组失败用例：所有模式忽略 VZ、PX4
   高度 profile 不符阻止启动、Gazebo/MAVROS AGL 持续分歧触发门控。
2. 实现单一 PX4 Z 位置闭环，删除外部 Z PD 参数和代码，运行新增及完整
   supervisor 测试。
3. 实现 MAVROS PX4 参数读取与起飞前 profile 验证，将期望值暴露为
   Gazebo supervisor 配置。
4. 实现 Gazebo 模型真值订阅、独立零点和持续误差门控；在起飞稳定判定、
   HOLD_READY 和 AUTO 中接入。
5. 修改 `gazebo-classic_iris_depth_camera` PX4 airframe 的高度融合和悬停推力默认值，
   检查 airframe 脚本语法和生效方式。
6. 运行 Python 回归测试、launch/config 结构测试、Python 语法检查和相关 catkin
   构建；审查 diff 只包含本轮直接变更。
7. 将新日志的根因、数据证据、修复内容、验证结果与下一次运行验收
   workflow 追加到 `RuntimeData/APEXNAV_MAJOR_ISSUES_20260818.md`。
