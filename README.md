第一阶段：链路跑通与“命名空间”化（核心：ROS 隔离）

关键任务：
        修改 Python 脚本：改造 habitat_vel_control.py，让它支持多 Agent。你需要利用 habitat_sim.AgentConfig 实例化两个机器人。

        话题隔离：使用 ROS Namespace。机器人 A 的所有话题前缀为 /robot1，机器人 B 为 /robot2。启动项配置：编写一个新的 launch 文件，同时启动两个 exploration_manager 实例，分别指向不同的命名空间。

具体操作：
        1.核心：修改 Python 桥接层 (real_world_test_habitat.py)
          修改要点：
            a.去除话题前导斜杠：确保所有订阅器（Subscriber）的话题名为相对路径，这样它们会自动继承 ROS Launch 文件中的 ns（命名空间）。
            b.引入机器人 ID 参数：用于区分不同的机器人实例。
        2.启动逻辑：修改 Launch 文件 (exploration_traj.launch)
          修改要点：
            a.使用 <group> 标签：为每个机器人建立独立的运行空间。
            b.重映射全局话题：如 /tf 和 /tf_static 是全局的，不需要动，但私有数据话题（odom, depth）必须区分。
        3.可视化逻辑： (exploration_fsm_traj.cpp)
          修改要点：
            a.Marker 命名空间隔离：RViz 根据 ns 和 id 区分 Marker。如果不改，Robot 2 的 Marker 会覆盖 Robot 1 的。
            b.Frame ID 处理：确保坐标系引用正确。
        4.地图与感知：map_ros.cpp & sdf_map2d.cpp
          修改要点：
            a.TF 监听器的 Frame ID：在多机环境下，camera_link 必须变为 b.robot_1/camera_link。参数读取：确保从私有句柄读取参数，避免不同机器人参数混淆。
        5. 坐标系树 (TF Tree) rviz_traj.launch
            修改要点：
            a.发布独立的静态变换：定义 world 到每个机器人的初始位置。






第二阶段：决策互斥（核心：实现互斥）

关键任务：
        引入位置交换：在 C++ 端的 ExplorationManager 中增加一个 ros::Subscriber，订阅对方的 Odometry。
        分值惩罚逻辑：在 findBestFrontier 循环中，计算每个 Frontier 到对方机器人的距离。
        效果：当对方离某个点更近时，你的机器人会自动将其视为“低价值”目标。导师建议：先做“距离互斥”，再做“语义互斥”。



第三阶段：轨迹解耦与记忆优化（核心：覆盖效率）
关键任务：
        路径惩罚：将对方走过的轨迹点存储在 std::vector 中，并在 A* 搜索时，给这些轨迹点附近的栅格增加额外的 Cost（代价），而不是设为障碍。这样机器人只有在万不得已时才会复用对方的路径。
        历史记录同步：将各自的 SDFMap 或 OccupancyGrid 进行简单的逻辑“或”运算合并，实现“我看到的你也知道”。导师建议：你可以直接修改 SDFMap2D::updateOccupancy，接收来自队友的点云数据。