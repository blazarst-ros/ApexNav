# Lite ApexNav 真机与Gazebo部署

正式感知链路只使用 **YOLOE（12184）+ CLIPITM（12182）**。完整的数据流、时间防御、PX4生命周期和验收要求见
[GAZEBO_REAL_WORLD_ROBUST_WORKFLOW.md](GAZEBO_REAL_WORLD_ROBUST_WORKFLOW.md)。

## 真机

真机始终使用相机自身标定的 `CameraInfo`，不会被HM3D内参覆盖。先启动相机、定位和TF，然后运行：

```bash
source devel/setup.bash
python real_world_test_example/real_world_tf_pose.py \
  _world_frame:=odom _camera_frame:=camera_color_optical_frame

roslaunch exploration_manager exploration_traj.launch \
  odom_topic:=/odom sensor_pose_topic:=/apexnav/camera/pose \
  depth_topic:=/camera/aligned_depth_to_color/image_raw \
  camera_info_topic:=/camera/color/camera_info \
  depth_unit_scale:=0.001 world_frame:=odom

python real_world_test_example/real_world_perception.py
```

`real_world_tf_pose.py` 在每一帧深度图的时间戳查询TF；查不到该时刻位姿就丢帧，不使用最新TF替代。

## Gazebo + PX4

下面采用分终端启动，便于逐层观察PX4、传感器、VLM、地图和控制。不要跳过每阶段的检查。

### 0. 只需执行一次：构建与依赖检查

```bash
cd /home/blazarst/ApexNav
source /opt/ros/noetic/setup.bash
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash

rospack find mavros
test -f /usr/share/GeographicLib/geoids/egm96-5.pgm
```

如果MAVROS缺失：

```bash
sudo apt update
sudo apt install ros-noetic-mavros ros-noetic-mavros-extras
sudo /opt/ros/noetic/lib/mavros/install_geographiclib_datasets.sh
```

### 1号终端：ROS Master

必须先启动ROS master，确保Gazebo相机插件初始化时就能连接ROS：

```bash
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11311
roscore
```

### 2号终端：small-house + PX4 Iris + RGB-D相机

这部分直接来自仓库根目录的 `gazebo_path.txt`：

```bash
export HOUSE=/media/blazarst/Getea/ApexGazebo/src/aws-robomaker-small-house-world
export PX4=/media/blazarst/Getea/ApexGazebo/src/PX4-Autopilot

source /usr/share/gazebo/setup.sh
source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11311
export GAZEBO_MODEL_PATH="$HOUSE/models:${GAZEBO_MODEL_PATH:-}"
export PX4_SITL_WORLD="$HOUSE/worlds/small_house.world"
rosparam set /use_sim_time true

cd "$PX4"
make px4_sitl gazebo-classic_iris_depth_camera
```

另开一个临时检查终端确认Gazebo数据存在：

```bash
source /opt/ros/noetic/setup.bash
rostopic hz /clock
rostopic hz /iris_depth_camera/camera/rgb/image_raw
rostopic hz /iris_depth_camera/camera/depth/image_raw
rostopic echo -n 1 /iris_depth_camera/camera/rgb/camera_info
```

RGB和深度应约为10Hz，深度通常为 `32FC1` 米制数据。

### 3号终端：MAVROS桥接PX4

```bash
source /opt/ros/noetic/setup.bash
source /home/blazarst/ApexNav/devel/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11311

roslaunch mavros px4.launch \
  fcu_url:="udp://:14540@localhost:14557"
```

检查飞控连接和ENU里程计：

```bash
rostopic echo -n 1 /mavros/state
rostopic hz /mavros/local_position/odom
```

`/mavros/state.connected` 必须为 `True`。QGroundControl可以另外启动用于观察，但进入自动任务后不要再用虚拟手柄或手动Arm覆盖监督器。

### 4号终端：Gazebo传感器与时间桥

```bash
source /opt/ros/noetic/setup.bash
source /home/blazarst/ApexNav/devel/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11311

roslaunch apexnav_gazebo gazebo_sensor_bridge.launch
```

该桥执行两件关键工作：

- 将MAVROS odom插值到每一帧深度图的源时间，发布 `/apexnav/camera/pose`。
- 将Gazebo原始相机重投影为HM3D-v2虚拟相机。

检查统一输出：

```bash
rostopic hz /apexnav/camera/rgb/image_raw
rostopic hz /apexnav/camera/depth/image_raw
rostopic hz /apexnav/camera/pose
rostopic echo -n 1 /apexnav/camera/camera_info
rostopic echo /apexnav/sensors/diagnostics
```

统一CameraInfo应为 `640x480`，K矩阵为：

```text
fx=388.1910413097385  fy=422.0475153598262
cx=320.0              cy=240.0
```

### 5号终端：YOLOE服务

```bash
cd /home/blazarst/ApexNav
source /home/blazarst/miniconda3/etc/profile.d/conda.sh
conda activate /media/blazarst/Getea/Lite-ApexNav/conda-env/Lite-apex
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m vlm.detector.yoloe --port 12184 \
  --weights /media/blazarst/Getea/Lite-ApexNav/model-cache/yoloe-11l-seg.pt
```

这里显式指定当前 Lite 环境中的 YOLOE 权重，避免 shell 中遗留的
`YOLOE_WEIGHTS` 指向旧的 `ApexData` 路径。代码会将该模型缓存目录作为后备搜索路径，
并复用其中的 Ultralytics `mobileclip_blt.ts`，首次预热不应再次下载该文件。

### 6号终端：CLIPITM服务

```bash
cd /home/blazarst/ApexNav
source /home/blazarst/miniconda3/etc/profile.d/conda.sh
conda activate /media/blazarst/Getea/Lite-ApexNav/conda-env/Lite-apex
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CLIP_DOWNLOAD_ROOT=/media/blazarst/Getea/Lite-ApexNav/model-cache/clip
python -m vlm.itm.clipitm --port 12182
```

该路径含有已校验的 `ViT-B-32.pt`；不要使用 `~/.cache/clip` 中下载中断留下的文件。

检查两个Lite模型，不应出现旧模型服务：

```bash
curl -s http://127.0.0.1:12184/healthz
curl -s http://127.0.0.1:12182/healthz
```

返回值中应分别包含 `"name":"yoloe"` 和 `"name":"clipitm"`，且 `ready=true`。

### 7号终端：Lite感知节点

```bash
cd /home/blazarst/ApexNav
source /home/blazarst/miniconda3/etc/profile.d/conda.sh
conda activate /media/blazarst/Getea/Lite-ApexNav/conda-env/Lite-apex
source /opt/ros/noetic/setup.bash
source devel/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11311
# 让ROS Noetic的cv_bridge使用与libp11-kit兼容的系统libffi ABI。
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7

python real_world_test_example/real_world_perception.py \
  --config-name gazebo_hm3dv2
```

任务开始前这里可能提示等待目标标签，这是正常的。模型健康状态应持续发布：

```bash
rostopic echo /apexnav/vlm/diagnostics
```

### 8号终端：地图、FSM、路径规划、轨迹与MPC

```bash
source /opt/ros/noetic/setup.bash
source /home/blazarst/ApexNav/devel/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11311

roslaunch apexnav_gazebo gazebo_planner.launch
```

此时规划器应停留在 `WAIT_TRIGGER`，还不会驱动PX4。检查：

```bash
rostopic echo /ros/state
rostopic hz /grid_map/occupied
```

### 9号终端：PX4生命周期与安全监督器

```bash
source /opt/ros/noetic/setup.bash
source /home/blazarst/ApexNav/devel/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11311

roslaunch apexnav_gazebo px4_supervisor.launch
```

监督器初始应处于 `WAIT_FCU=0`，不会自行Arm：

```bash
rostopic echo /apexnav/mission/state
```

### 10号终端：RViz与VLM证据

```bash
source /opt/ros/noetic/setup.bash
source /home/blazarst/ApexNav/devel/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11311

roslaunch exploration_manager rviz_traj.launch fixed_frame:=map
```

RViz中可观察HM3D统一图像、YOLOE标注图、VLM状态文字、占据地图、ObjectMap、ValueMap、frontier和轨迹。

### 11号终端：启动chair任务

先完成最终预检：

```bash
source /opt/ros/noetic/setup.bash
source /home/blazarst/ApexNav/devel/setup.bash
export ROS_MASTER_URI=http://127.0.0.1:11311

rostopic echo -n 1 /mavros/state
rostopic echo -n 1 /apexnav/sensors/diagnostics
rostopic echo -n 1 /apexnav/vlm/diagnostics
rostopic echo -n 1 /apexnav/camera/camera_info
```

然后提交任务：

```bash
rosservice call /apexnav/mission/start "target_label: 'chair'"
```

监督器随后自动执行：

```text
PRESTREAM -> ARM -> OFFBOARD_TAKEOFF -> HOLD_READY
          -> AUTO -> HOLD/FAULT -> AUTO.LAND -> DISARMED
```

持续监控：

```bash
rostopic echo /apexnav/mission/state
rostopic hz /mavros/setpoint_raw/local
rostopic hz /apexnav/planner/cmd_vel_raw
rqt_image_view /apexnav/vlm/annotated_image
```

### 安全停止和关闭顺序

任何时候优先调用：

```bash
rosservice call /apexnav/mission/stop
```

等待 `/apexnav/mission/state.state` 变为 `DISARMED=8`，再依次Ctrl-C关闭：

1. PX4监督器。
2. 规划器。
3. 感知节点和RViz。
4. 传感器桥。
5. MAVROS。
6. PX4/Gazebo。
7. YOLOE、CLIPITM和roscore。

不要直接先杀PX4、MAVROS或监督器，否则无法保证受控降落。

### 常见阻塞定位

- 没有原始相机话题：检查PX4 target是否为 `gazebo-classic_iris_depth_camera`，并确认roscore先于Gazebo启动。
- MAVROS未连接：检查UDP端口、PX4 SITL是否仍运行，以及 `/mavros/state`。
- 统一相机无输出：检查原始CameraInfo、MAVROS odom和 `/apexnav/sensors/diagnostics` 的丢帧原因。
- 感知无输出：先检查两个 `/healthz`，再检查目标标签和 `/apexnav/vlm/diagnostics`。
- 地图无输出：核对depth、camera pose、CameraInfo频率和时间戳；MapROS拒绝超过10ms的depth/pose组合。
- 任务不Arm：监督器只有在FCU、odom、RGB、Depth均新鲜，并收到当前目标的、新鲜且
  YOLOE/CLIPITM均有效的 `SemanticObservation` 后才进入PRESTREAM；这是一项刻意的安全门禁。

`gazebo_full_stack.launch` 仍保留用于最终集成回归；学习和调试时优先使用上述分终端流程。
