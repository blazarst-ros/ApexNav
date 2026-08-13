# Branch: I2S1

**Functions:** Selection of the height geometric indicator function and determination of the coefficients

**Update to:** `049971b613fcd2554b034bd4a853d79cc1e61613`

**Not included:** Experimental data

**Feature:** Multi-agent

**Language prior:** Added

**System:** Height heterogeneous system

## Runtime Exploration Settings

This branch runs the multi-agent evaluation with heterogeneous camera heights:

| Agent | Height | RGB sensor y | Depth sensor y |
| --- | ---: | ---: | ---: |
| `agent_0` | `0.7 m` | `0.7 m` | `0.7 m` |
| `agent_1` | `1.2 m` | `1.2 m` | `1.2 m` |

The evaluation success distance is:

```text
0.35 m
```

Config key:

```yaml
success_distance: 0.35
```

The maximum episode length for evaluation is:

```yaml
max_episode_steps: 400
```

Step counting policy:

```text
Every executed Habitat action increments the per-agent step counter.
```

This matches the original single-agent counting semantics: forward motion,
turning, looking up/down, and stop actions all count as one step for the agent
that executed the action.

Runtime progress logging prints each agent's own step counter, for example:

```text
--------------Steps: agent_0=288, agent_1=286--------------
```

The C++ planner treats a searched object as reached at:

```text
REACH_DISTANCE = 0.20 m
SOFT_REACH_DISTANCE = 0.20 m
```

The C++ planner reach-claim thresholds remain tighter than the Python/Habitat
evaluation success distance; the stuck-recovery branch no longer relaxes the
C++ reach-claim criterion.

Multi-agent perception scheduling:

```yaml
perception_agents_per_step: 2
perception_interval_steps: 1
```

This allows both agents to run ITM/VLM object detection in the same
simulation loop when both have changed viewpoint. `perception_interval_steps`
is counted per agent, not as a global scheduler interval.

These values are applied consistently in:

- `config/habitat_eval_hm3dv1.yaml`
- `config/habitat_eval_hm3dv2.yaml`
- `config/habitat_eval_mp3d.yaml`

`habitat2ros.ROSPublisher` receives the configured camera height from
`habitat_evaluation.py`, so the ROS sensor pose z-offset matches the Habitat
RGB/depth sensor height.

Stage 1 detection records are published on:

```text
/stage1/detector/detection
```

and written as JSONL files under:

```text
/media/blazarst/Getea/RuntimeData/Stage1_detector
```

## Multi-Agent Exploration State and Termination

## 当前 Frontier 规划与多 Agent 协调算法

本节描述**当前代码实际运行的算法**。其中，双 Agent 在同一规划周期、
最高优先级模式一致时，会进入双起点 `MINMAX` 联合分配；其余情况保持独立规划。

### 1. Frontier 的生成与失效

规划器在共享二维占据栅格中将“未知栅格且相邻自由栅格”的位置识别为
Frontier。每轮地图更新后，它会：

1. 在局部更新区域扫描 Frontier 种子；
2. 使用 BFS 将连通种子扩展为 Frontier 簇；
3. 丢弃过小簇；
4. 对超过 `frontier/cluster_size_xy` 的长簇沿 PCA 第一主方向递归切分；
5. 使用每个簇的几何中心作为候选目标。

Frontier 若已被观察完成、相连未知区域不足，或机器人长时间无法到达，
会转为 dormant 或被移除。候选目标必须通过 A* 可达性检查。

### 2. 单 Agent 的目标优先级

每台 Agent 目前独立按以下优先级选择下一个任务：

```text
SEARCH_BEST_OBJECT
→ SEARCH_OVER_DEPTH_OBJECT
→ 活跃 Frontier 探索
→ SEARCH_SUSPICIOUS_OBJECT
→ dormant Frontier
→ 极端回退搜索
```

只有当前不存在可执行的高优先级对象任务时，系统才会选择 Frontier。
因此，语义 Frontier 不是“已发现目标后的导航”，而是在尚未获得可直接
导航对象时，根据图文匹配语义线索选择探索方向。

### 3. 当前 Hybrid Frontier 策略

默认参数 `exploration/policy = 2`，即 Hybrid：

```text
语义差异显著 → 选择高语义 Frontier，并用单 Agent ATSP 排序
语义差异不显著 → 选择 A* 实际路径最短的 Frontier
```

每个 Frontier 的语义价值取 Value Map 中其周围 `5×5` 邻域的最大值，并
跳过障碍物和膨胀障碍栅格。仅保留 A* 可达的 Frontier。若语义值标准差
超过 `exploration/sigma_threshold`，且最大值/均值超过
`exploration/max_to_mean_threshold`，则进入语义利用模式；否则进入几何
探索模式。

几何模式比较的是 A* 路径长度，不是欧氏直线距离。当前 A* 直接使用占据
栅格和膨胀占据栅格判断安全性；ESDF 距离场由共享 SDFMap2D 维护，但尚未
直接加入 A* 的路径代价。

### 4. 当前 LKH-ATSP 的实际作用

在语义利用模式中，系统为**当前正在规划的单台 Agent**构造：

```text
当前 Agent 位置 → 高语义 Frontier
高语义 Frontier → 高语义 Frontier
```

的 A* 路径长度矩阵，并调用 LKH 排序。当前参数文件固定写入：

```text
SALESMEN = 1
MTSP_OBJECTIVE = MINSUM
```

因此，这里是单 Agent 的 ATSP/开放路线排序器：系统从返回顺序中只取第一
个可达 Frontier 作为当前目标，随后继续滚动重规划。尽管仓库内的
`lkh_mtsp_solver` 库具备 `SALESMEN > 1` 的 MTSP 能力，当前 ApexNav 并**没有**
运行多旅行商协同规划。

### 5. 当前多 Agent 协调方式

当前有两台 Agent，共享地图、对象图、Value Map 和 Frontier 列表，但仍按
Agent 顺序调用单 Agent 规划。正常流程为：

```text
Agent 0 独立选择目标并写入 Frontier Claim
→ Agent 1 规划时优先过滤被 Agent 0 Claim 的 Frontier
→ 若所有 Frontier 都被 Claim，则退化为允许共享选择
```

这是一种顺序相关的贪心软互斥机制，不是集中式全局分配，也不保证两台
Agent 的路线负载均衡。

### 6. 双 Agent 模式门控联合分配

仅在两个 Agent 都需要重新规划且最高优先级模式一致时，建立该模式
专属的共享候选池。例如：

```text
两个 SEARCH_BEST_OBJECT → 仅高置信度对象簇
两个 SEARCH_OVER_DEPTH_OBJECT → 仅过深对象簇
两个 Semantic Frontier → 仅通过语义门控的活跃 Frontier
两个 Geometric Frontier → 仅活跃且可达的 Frontier
```

不同模式继续独立规划，因为它们的目标类别不同，不需要 Claim。对于共同
模式，将使用两个不同起点 `R0`、`R1` 的 A* 距离，求解：

```text
min max(L0, L1)
```

即最小化两条路线中的较长路线。现有 LKH MTSP 封装只支持单一 depot，不能
正确表示 `R0`、`R1` 两个不同起点；因此不会仅通过将 `SALESMEN` 改为 `2`
来启用它。当前实现采用候选上限 `K=10` 内的双起点 Held--Karp 动态规划，
对两个互补候选子集选择 `min max(L0, L1)` 的分配；每轮仍只下发每台 Agent
路线的首个目标，并在下一轮滚动重规划。少于两个共同可达候选时不进入联合分配，
由原独立规划路径处理。

The planner keeps a separate control FSM for each configured agent and publishes
the current ROS planner state array on:

```text
/ros/state_all
```

The fine-grained exploration target state (`EXPL_RESULT`) is also tracked per
agent and published on:

```text
/ros/expl_result_all
/ros/agent_0/expl_result
/ros/agent_1/expl_result
```

The message type is `std_msgs/Int32MultiArray`, where `data[i]` is the latest
`EXPL_RESULT` for `agent_i`. The per-agent topics are `std_msgs/Int32` and are
easier to inspect with `rostopic echo`. The existing scalar `/ros/expl_result`
topic is kept for compatibility and still reflects the most recently published
exploration result.

`EXPL_RESULT` values are:

```text
0 EXPLORATION
1 SEARCH_BEST_OBJECT
2 SEARCH_OVER_DEPTH_OBJECT
3 SEARCH_SUSPICIOUS_OBJECT
4 NO_PASSABLE_FRONTIER
5 NO_COVERABLE_FRONTIER
6 SEARCH_EXTREME
```

In `episode_termination: cooperative`, a successful target claim remains a
global stop condition: if any agent reaches `FINAL_RESULT.REACH_OBJECT`, the C++
planner broadcasts `STOP`, moves all agents to `FINISH`, and Python evaluates
the episode outcome. Planner failures use the separate `FINISH_FAILURE` state:
if one agent is stuck or has no usable frontier, only that agent stops while the
other agents continue exploring against the shared map. The episode ends as an
overall planner failure only after every agent reaches `FINISH_FAILURE` (with
the Habitat step budget retained as an independent hard timeout).

Final multi-agent success is still evaluated by Python after episode shutdown:
the episode is counted as successful if any agent satisfies the ObjectNav stop
success condition.

## Multi-Agent Episode Reset Fix

The multi-agent branch can keep ROS subscribers and timers alive across
episodes while only resetting internal planner state. This differs from the
single-agent `main` branch, which rebuilds the planner interface on episode
finish. In the multi-agent path, stale per-agent buffers and concurrent writes
to the shared map can therefore survive into the next episode.

The original observed failure mode was a C++ `exploration_node` crash after
episode reset, followed by Python repeatedly waiting in `WAIT_ACTION_FINISH`.
The Python wait state is a downstream symptom: planner state feedback becomes
stale after the C++ node exits.

The overstep path has a separate reset-handshake failure mode. When an episode
ends by reaching `max_episode_steps`, Python publishes `EPISODE_FINISH` and
waits for `/ros/state_all` to confirm that the C++ planner has reset. C++ map
reset can take several seconds, so the old 3-second Python acknowledgement
window could expire before the reset state was published. Python then sent a
second `EPISODE_FINISH`, causing another map reset, and finally misreported the
blocked state feedback as a stale planner process.

This branch now resets episode-owned planner state explicitly:

- `MapROS::resetEpisodeState()` clears all per-agent camera, depth, object, ITM,
  over-depth, and shared map state under `map_mutex_`.
- `SDFMap2D::resetMap()` routes through `MapROS` so sensor callbacks cannot
  access map buffers while reset replaces them.
- Per-agent depth clouds are re-preallocated after reset to avoid indexed writes
  into an empty point cloud.
- Per-frame virtual-ground buffers are cleared before each depth update.
- Object visualization uses bounded object/label counts to avoid out-of-range
  access when semantic labels lag object geometry.
- The over-depth object cache is an `ExplorationManager` member and is cleared
  between episodes instead of remaining as a function-local static.
- `/ros/state_all` is republished after FSM transitions so Python observes the
  current multi-agent state.
- `/ros/state_all` is also published immediately after `resetEpisode()` finishes
  so Python receives an explicit reset acknowledgement.
- `habitat_evaluation.py` uses a longer reset acknowledgement window and a
  reset-specific stale timeout, preventing overstep cleanup from retriggering
  repeated C++ map resets.
- `habitat_evaluation.py` now fails fast if planner state feedback is stale,
  making a dead `exploration_node` visible instead of masking it as an action
  wait loop.
- ATSP tour parsing now guards invalid or out-of-range solver output.

Validation used for this change:

```bash
python3 -c "import importlib.util; spec=importlib.util.spec_from_file_location('t','tests/test_episode_reset_lifecycle.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); tests=[getattr(m,n) for n in dir(m) if n.startswith('test_')]; [test() for test in tests]; print(f'{len(tests)} episode-reset regression checks passed')"
python3 -m py_compile habitat_evaluation.py tests/test_episode_reset_lifecycle.py
source /opt/ros/noetic/setup.bash && catkin_make --pkg exploration_manager -j2
git diff --check
```

## Semantic Object Visualization

This branch adds RViz visualization for object-centric semantic fusion. The
visualization separates regular grid occupancy from semantic object clusters and
keeps the cluster status table stable across updates.

### RViz Topics

| Content | ROS topic | Type | Display |
| --- | --- | --- | --- |
| Regular occupied grid | `/grid_map/occupied` | `sensor_msgs/PointCloud2` | Dark gray obstacles |
| Inflated occupied grid | `/grid_map/occupied_inflate` | `sensor_msgs/PointCloud2` | Gray inflated obstacles |
| Legacy object occupancy | `/grid_map/occupancy_object` | `sensor_msgs/PointCloud2` | Kept for compatibility, disabled in RViz |
| Semantic object grid | `/grid_map/semantic_objects` | `sensor_msgs/PointCloud2` with RGB | Enabled by default |
| Cluster ID labels | `/object/cluster_markers` | `visualization_msgs/MarkerArray` | Enabled by default |
| Structured cluster status | `/object/cluster_status` | `plan_env/ObjectClusterStatusArray` | For scripts and debugging |
| RViz status table | `/object/cluster_status_image` | `sensor_msgs/Image` | Enabled as an Image display |

The semantic object grid uses fixed colors:

| Cluster state | Condition | Color |
| --- | --- | --- |
| Target | `best_label == 0` | Red |
| Confusion object | `best_label > 0` | Cyan |
| Uncertain | `best_label == -1` | Gray |
| Ordinary obstacle | Not an object cluster | Dark gray from `/grid_map/occupied` |

The semantic grid is published slightly above the obstacle grid so target and
confusion clusters remain visible when they overlap occupied cells.

### Cluster IDs and Competition Rule

Object cluster IDs are unique within one episode and are assigned in creation
order: `C000`, `C001`, `C002`, and so on. IDs only apply to semantic object
clusters produced by target or confusion detections. Ordinary obstacle cells do
not receive cluster IDs.

The map color and marker state follow the existing object-centric fusion rule:

```text
best_label = argmax(evidence_points[label] * fused_confidence[label])
```

If the same physical area receives competing labels, for example chair and
table evidence projecting to the same cluster, the cluster is not duplicated.
The current winning label determines whether the cluster is displayed as target
red, confusion cyan, or uncertain gray. The cluster ID and the row position in
the status table remain stable even when the winning label changes.

### Status Table

`/object/cluster_status_image` is a generated image table for RViz. Add it with
an RViz `Image` display, or use the included `ApexNav.rviz` and
`ApexNav_Traj.rviz` configs where it is already enabled.

Each cluster occupies one or more consecutive rows:

```text
cluster | label | obs | score
```

Clusters are always ordered by cluster ID. Every configured candidate label is
shown on its own row within the cluster, including labels whose observation
count is currently zero. Existing clusters only update their numeric values;
they are not resorted by score. New clusters append to the bottom of the table.
This keeps the display stable during live simulation.

The structured `/object/cluster_status` topic contains all candidate labels for
each cluster. The image table uses only the numeric label index, observation
count, and competition score. The map marker is also compact, for example
`0 obs=4 score=102.316`; cluster state remains visible through marker color.

Field meanings:

- `cloud_points`: current voxel-downsampled points stored in the cluster for
  that label.
- `evidence_points`: accumulated evidence used by fusion. Under Fusion Type 1,
  this can also include negative evidence from visible-but-undetected objects,
  so it is not always equal to accumulated detected point cloud size.
- `detection_count`: number of positive detections fused into that label.
- `fused_confidence`: current fused confidence for that label.
- `score`: `evidence_points * fused_confidence`, the score used for label
  competition.

### Detection Class Names

The detection message now carries the class dictionary:

```text
MultipleMasksWithConfidence.class_names
```

`class_names[0]` is the target category. `class_names[1...]` are the similar or
confusion categories. Detection results still use `label_indices[i]` to refer to
this dictionary, so the existing convention remains:

```text
label 0 = target object
label > 0 = confusion object
```

The C++ object map expands label storage dynamically from this dictionary, so
more than five candidate categories can be visualized and fused safely.

### Usage

Build the ROS workspace after changing message definitions:

```bash
cd ~/ApexNav
catkin_make
source devel/setup.bash
```

Then start Habitat, the ROS planner/map node, and RViz with either project
config:

```bash
rviz -d src/planner/exploration_manager/config/ApexNav.rviz
rviz -d src/planner/exploration_manager/config/ApexNav_Traj.rviz
```

No extra visualization script is required. Running Habitat alone only shows the
camera-side detection output; the accumulated clusters, semantic grid, marker
IDs, and fusion table require the ROS map/planner node.

Useful runtime checks:

```bash
rostopic echo -n 1 /object/cluster_status
rostopic echo -n 1 /grid_map/semantic_objects/width
rostopic echo -n 1 /object/clouds/width
```

### Episode Reset Behavior

On episode finish, `SDFMap2D::resetMap()` routes through
`MapROS::resetEpisodeState()`. The reset now publishes explicit empty
replacement clouds and marker deletion messages so RViz does not keep stale
data when `Decay Time` is zero.

The reset clears or replaces:

- semantic object grid `/grid_map/semantic_objects`
- cluster markers `/object/cluster_markers`
- cluster status table `/object/cluster_status_image`
- structured cluster status `/object/cluster_status`
- legacy object cloud `/object/clouds`
- filtered and over-depth object clouds
- occupied, inflated, unknown, free, ESDF, object, and value-map grid displays

This clearing depends on the normal episode-finish handshake reaching the ROS
planner. After a successful reset, new cluster IDs start again from `C000`.
