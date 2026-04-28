# Dataflow: Simulator to Decision Making

This note traces the runtime dataflow of one UAV from simulation outputs to the exploration decision made by `murder_swarm`, then adds the swarm-side dataflow that affects that decision.

## 1. Launch-Level Wiring

The typical maze entrypoint is:

- `Exploration/murder_swarm/launch/murder_swarm_maze4.launch`

That launch file starts:

- Gazebo with a maze world
- the communication bridge
- one `ground_node`
- one `murder_swarm_node` per UAV through `murder_single.launch`
- one `traj_exc_node` per UAV

Inside `murder_single.launch`, simulator topics are remapped into the names expected by `murder_swarm_node`:

- `/depth` <- `/firefly$(arg id)/vi_sensor/camera_depth/depth/disparity`
- `/pointcloud` <- `/firefly$(arg id)/vi_sensor/camera_depth/depth/points`
- `/odom` <- `/firefly$(arg id)/ground_truth/odometry`
- `/vi_odom` <- `/firefly$(arg id)/vi_sensor/ground_truth/odometry`
- `/block_map/caminfo` <- `/firefly$(arg id)/vi_sensor/camera_depth/depth/camera_info`

So the simulator-side data sources are RotorS/Gazebo sensor topics, but the planner always sees the normalized names above.

## 2. High-Level Pipeline

For one UAV, the main runtime path is:

```text
Gazebo / RotorS
  -> depth image or point cloud
  -> VI odometry
  -> body odometry
  -> murder_swarm::Murder
  -> BlockMap
  -> FrontierGrid
  -> MultiDTG
  -> GraphVoronoiPartition
  -> MurderFSM
  -> LocalPlan / GlobalPlan decision
  -> /Murder/Traj
```

There are two loops running in parallel:

1. A fast perception/update loop driven by synchronized sensor + `vi_odom`
2. A decision loop driven by `MurderFSM`

## 3. Perception and State Update Flow

### 3.1 Node Initialization

`Murder::init()` wires the modules in this order:

- `SwarmDataManager`
- `BlockMap`
- `ColorManager`
- `LowResMap`
- trajectory optimizer `TrajOpt_`
- `FrontierGrid`
- `MultiDTG`
- `GraphVoronoiPartition`
- `YawPlanner`

This is the dependency chain:

```text
SwarmDataManager
BlockMap -> LowResMap -> FrontierGrid -> MultiDTG -> GraphVoronoiPartition
                                     \-> Murder planner
Trajectory optimizer + YawPlanner ---/
```

### 3.2 Body Odometry Path

`/odom` is subscribed by `Murder::BodyOdomCallback()`.

This callback updates:

- current position `p_`
- current velocity `v_`
- current yaw `yaw_`
- full robot pose matrix `robot_pose_`
- `FG_.Robot_pos_`
- `SwarmDataManager::SetPose(*odom)`

This body odometry is the planner's current state estimate for decision making and for broadcasting pose to the swarm.

### 3.3 Sensor + VI Odom Path

Depending on config:

- depth mode: `/depth` + `/vi_odom` -> `ImgOdomCallback()`
- point-cloud mode: `/pointcloud` + `/vi_odom` -> `PCLOdomCallback()`

In both callbacks, the order is:

1. `BM_.OdomCallback(odom)`
2. `BM_.InsertImg(img)` or `BM_.InsertPcl(pcl)`
3. `FG_.UpdateFrontier(BM_.newly_register_idx_)`
4. `MDTG_.Update(robot_pose_, !sensor_flag_)`
5. set `sensor_flag_ = true`

This is the core online map-building pipeline.

## 4. What Each Module Produces

### 4.1 BlockMap

`BlockMap` converts depth / point cloud observations into a voxel occupancy map.

Main outputs used by the planner:

- `cur_pcl_`: current occupied points used by `LowResMap`
- `newly_register_idx_`: newly observed voxels used by `FrontierGrid`
- occupancy queries such as:
  - `GetVoxState()`
  - `PosBBXFree()`
  - `PosBBXOccupied()`

Conceptually:

```text
sensor + vi_odom
  -> camera/body pose in world
  -> mark free / occupied voxels
  -> produce newly observed cells
```

### 4.2 FrontierGrid

`FrontierGrid::UpdateFrontier()` consumes `BlockMap::newly_register_idx_`.

Its job is to:

- update frontier cell states
- mark newly explorable frontiers
- mark fully explored frontiers
- maintain candidate viewpoints for each frontier

Later, planning may call:

- `SampleVps()`
- `StrongCheckViewpoint()`
- `GetVp()` / `GetVpPos()`

So `FrontierGrid` is where "unknown boundary" becomes "candidate exploration target".

### 4.3 MultiDTG

`MultiDTG::Update()` is called after every sensor-map update.

Its job is to combine:

- current low-resolution traversability from `LowResMap`
- local frontier/viewpoint candidates from `FrontierGrid`
- current robot pose
- existing graph structure

to maintain the MR-DTG:

- H-nodes: topological free-space anchor nodes
- F-nodes: frontier nodes
- HH edges: topology-to-topology connectivity
- HF edges: topology-to-frontier connectivity

It also populates local working sets such as:

- `local_h_list_`
- `local_f_list_`
- `local_h_dist_list_`
- `local_f_dist_list_`

These are later consumed by graph partition and target selection.

### 4.4 GraphVoronoiPartition

`GraphVoronoiPartition` runs on timers, not only when a plan is requested.

Background timers do three things:

- `PartitionTimerCallback()`
  - runs `LocalGVP()` and `GlobalGVP()`
- `StateTimerCallback()`
  - sends local partition / connectivity state to the swarm
- `JobTimerCallback()`
  - receives other UAVs' current jobs

Its role is to turn DTG structure into ownership and priority:

- which frontiers are locally assigned to this UAV
- which topological regions are globally assigned
- which target is better given other UAVs' claimed jobs and distances

So this is the main module where "available targets" become "my targets".

## 5. Decision-Making Flow in `MurderFSM`

`MurderFSM` runs every 0.01 s.

It gates planning through `Murder::AllowPlan()`:

- plan interval reached
- current position feasible
- fresh sensor update available
- viewpoints sampled

FSM states:

- `SLEEP`
- `LOCALPLAN`
- `GLOBALPLAN`
- `EXCUTE`
- `FINISH`

Decision logic:

```text
SLEEP
  -> after /start_trigger -> LOCALPLAN

LOCALPLAN
  -> if local target found and trajectory succeeds -> EXCUTE
  -> else if local targets are weak / missing -> GLOBALPLAN or FINISH

GLOBALPLAN
  -> if global target found and trajectory succeeds -> EXCUTE
  -> else -> LOCALPLAN or FINISH

EXCUTE
  -> keep checking trajectory, viewpoints, and replanning time
  -> on failure -> LOCALPLAN
```

## 6. How a Local Decision Is Made

When `LOCALPLAN` runs:

1. If viewpoints are not sampled yet, call `FG_.SampleVps()`
2. Call `GVP_.GetLocalFNodes(...)`
3. This searches local frontier viewpoints using:
   - `FrontierGrid`
   - local DTG connectivity
   - local path distance from `LowResMap`
   - swarm partition ownership from `GraphVoronoiPartition`
4. The planner gets:
   - target frontier/viewpoint `f_v`
   - path cost
   - target state `t_state`
   - whether the path is free, dangerous, or unavailable
5. `Murder::LocalPlan()` converts that target into a trajectory with `TrajPlanB(...)`
6. If successful, publish the trajectory and record the current job in `GVP_`

This is the local decision path:

```text
local frontier candidates
  + local traversability
  + local DTG links
  + current swarm partition
  -> GetLocalFNodes()
  -> choose frontier/viewpoint
  -> trajectory optimization
  -> decision = execute this local target
```

## 7. How a Global Decision Is Made

When `GLOBALPLAN` runs:

1. Call `GVP_.GetGlobalFNodes(...)`
2. That function:
   - searches nearby H-nodes with `LowResMap`
   - creates a fake start node for the current UAV
   - searches global DTG connectivity through `MultiDTG`
   - evaluates candidate targets with `GetBestTarget(...)`
3. `GetBestTarget(...)` scores targets using:
   - number of reachable frontier opportunities behind an H-node
   - travel distance
   - other UAVs' active jobs
   - partition penalties and decay terms
4. The chosen target viewpoint is revalidated with `FG_.StrongCheckViewpoint(...)`
5. `Murder::GlobalPlan()` then runs trajectory generation
6. On success, publish the trajectory and mark the current global job

This is the global decision path:

```text
global DTG
  + local-to-global connection from current pose
  + frontier gains behind candidate H-nodes
  + swarm job conflicts
  -> GetGlobalFNodes()
  -> GetBestTarget()
  -> choose H-node / frontier / viewpoint
  -> trajectory optimization
  -> decision = execute this global target
```

## 8. Swarm Dataflow That Affects Decisions

The decision is not purely local. `SwarmDataManager` continuously exchanges:

- poses
- trajectories
- DTG nodes / edges
- frontier state
- current jobs
- partition/connectivity state
- map blocks

For decision making, the most important shared data are:

- pose updates from `SetPose()`
- DTG updates from `SetDTGHn()`, `SetDTGHFEdge()`, `SetDTGHHEdge()`
- frontier status from `SetDTGFn()`
- partition/connectivity state from `SetState()`
- job announcements from `SetJob()`

The practical effect is:

```text
other UAV states/jobs/DTG
  -> SwarmDataManager receive buffers
  -> GraphVoronoiPartition::LoadSwarmJob / LoadSwarmState
  -> ownership and gain updates
  -> changes which target this UAV selects
```

## 9. End-to-End Dataflow Summary

### 9.1 Main Online Path

```text
Gazebo / RotorS
  -> /fireflyN/.../depth or /points
  -> /fireflyN/.../ground_truth/odometry
  -> remap to /depth or /pointcloud, /vi_odom, /odom
  -> Murder callbacks
  -> BlockMap occupancy update
  -> FrontierGrid frontier update
  -> MultiDTG topology update
  -> GraphVoronoiPartition background partition update
  -> MurderFSM checks if planning is allowed
  -> LocalPlan or GlobalPlan
  -> trajectory decision published on /Murder/Traj
```

### 9.2 Decision Inputs at the Moment of Planning

At planning time, `MurderFSM` is effectively consuming these state products:

- current UAV state from `BodyOdomCallback`
- current occupancy / feasibility from `BlockMap` and `LowResMap`
- current frontier/viewpoint candidates from `FrontierGrid`
- current topological graph from `MultiDTG`
- current partition and swarm-job context from `GraphVoronoiPartition` and `SwarmDataManager`

That combination is the actual "decision state" of the system.

## 10. Short Version

If you want the shortest accurate description:

```text
Simulator sensors/odom
  -> BlockMap builds occupancy
  -> FrontierGrid extracts frontier targets
  -> MultiDTG builds local/global topological connectivity
  -> GraphVoronoiPartition allocates targets under swarm context
  -> MurderFSM chooses local/global planning mode
  -> Murder generates the trajectory for the selected target
```

## 11. Main Source Files

- `Exploration/murder_swarm/src/murder.cpp`
- `Exploration/murder_swarm/src/murderFSM.cpp`
- `Mapping/block_map/src/block_map.cpp`
- `Mapping/frontier_grid/src/frontier_grid.cpp`
- `Exploration/multiDTG/src/multiDTG.cpp`
- `Exploration/graph_partition/src/graph_partition.cpp`
- `Communication/swarm_data/src/swarm_data.cpp`
- `Exploration/murder_swarm/launch/murder_single.launch`
- `Exploration/murder_swarm/launch/murder_swarm_maze4.launch`
