# Navigation System Examination Report

Date: 2026-05-05

Scope: static and offline validation of the current Habitat evaluation, ROS FSM handshake, VLM topic wiring, and configuration defaults. A full runtime reliability result still requires a live ROS + Habitat stress run.

## Executive Summary

Current status: **partially validated**.

The Python scripts compile, the Python/C++ state enum values match, the main ROS topics are wired consistently, success distance is set to `0.3`, and the startup trigger deadlock fix is present. The system is ready for a controlled live run, but long-run reliability cannot be claimed from static checks alone.

Main remaining risk: **runtime handshake reliability under repeated episodes**. The code now has safeguards for early triggers and missed episode reset acknowledgements, but this must be verified with 10/50/100 episode runs.

## Checks Performed

### 1. Python Syntax

Command:

```bash
python3 -m py_compile habitat_evaluation.py real_world_test_example/real_world_test_habitat.py habitat_vel_control.py habitat_manual_control.py habitat_manual_control_multiagent.py
```

Result: **PASS**

No Python syntax errors were found in the checked navigation/evaluation scripts.

### 2. State Enum Consistency

Checked files:

- `params.py`
- `src/planner/exploration_manager/include/exploration_manager/exploration_fsm.h`

Result: **PASS**

Python and C++ define matching state/action integer values.

Habitat/Python state:

| Value | State |
|---:|---|
| `0` | `READY` |
| `1` | `ACTION_EXEC` |
| `2` | `ACTION_FINISH` |
| `3` | `EPISODE_FINISH` |

C++ ROS FSM state:

| Value | State |
|---:|---|
| `0` | `INIT` |
| `1` | `WAIT_TRIGGER` |
| `2` | `PLAN_ACTION` |
| `3` | `WAIT_ACTION_FINISH` |
| `4` | `PUB_ACTION` |
| `5` | `FINISH` |

Action values:

| Value | Action |
|---:|---|
| `0` | `STOP` |
| `1` | `MOVE_FORWARD` |
| `2` | `TURN_LEFT` |
| `3` | `TURN_RIGHT` |
| `4` | `TURN_DOWN` |
| `5` | `TURN_UP` |

### 3. Python-C++ Handshake Wiring

Checked files:

- `habitat_evaluation.py`
- `src/planner/exploration_manager/src/exploration_fsm.cpp`

Result: **PASS, static**

Relevant topic wiring:

| Direction | Topic | Message | Purpose |
|---|---|---|---|
| C++ -> Python | `/ros/state_all` | `Int32MultiArray` | Per-agent FSM states |
| Python -> C++ | `/habitat/state` | `Int32` | Habitat action/episode state |
| Python -> C++ | `/move_base_simple/goal` | `PoseStamped` | Trigger C++ from `WAIT_TRIGGER` |
| C++ -> Python | per-agent action topics | `Int32` | Planner action commands |
| Python -> C++ | `/detector/confidence_threshold` | `Float64` | Object confidence threshold |

Important current safeguards found in `habitat_evaluation.py`:

- The readiness timer publishes odom/depth/RGB/confidence only.
- It no longer publishes `/move_base_simple/goal` before all agents reach `WAIT_TRIGGER`.
- If Python sees `WAIT_ACTION_FINISH` before it intentionally triggers, it calls `_finish_episode_handshake()` to recover.
- `_finish_episode_handshake()` sends `EPISODE_FINISH`, waits for C++ acknowledgement through `/ros/state_all`, and retries slowly instead of flooding reset messages.

### 4. C++ FSM State Flow

Checked file:

- `src/planner/exploration_manager/src/exploration_fsm.cpp`

Result: **PASS, static**

Observed workflow:

```text
INIT
  waits for odom + confidence
  |
  v
WAIT_TRIGGER
  waits for Python trigger
  |
  v
PLAN_ACTION
  decides action
  |
  v
PUB_ACTION
  publishes action
  |
  v
WAIT_ACTION_FINISH
  waits for Python ACTION_FINISH
  |
  v
PLAN_ACTION
```

Episode reset:

```text
Python EPISODE_FINISH
  -> C++ habitatStateCallback()
  -> resetEpisode()
  -> all agents INIT
  -> wait for odom/confidence
  -> WAIT_TRIGGER
```

Runtime risk: if Python does not publish `ACTION_FINISH`, C++ remains in `WAIT_ACTION_FINISH` and republishes the last action after timeout.

### 5. Configuration Defaults

Checked files:

- `config/habitat_eval_hm3dv1.yaml`
- `config/habitat_eval_hm3dv2.yaml`
- `config/habitat_eval_mp3d.yaml`
- `config/habitat_vel_control.yaml`
- `real_world_test_example/config/real_world_test.yaml`

Result: **PASS**

Current relevant values:

| Config | Value |
|---|---|
| `success_distance` | `0.3` |
| `num_agents` in main eval configs | `3` |
| `perception_agents_per_step` | `1` |
| `perception_interval_steps` | `1` |
| YOLO confidence | `0.3` |
| GroundingDINO confidence in eval configs | `0.40` |

Interpretation:

- Success is evaluated at `0.3 m`.
- Only one agent is budgeted for perception per scheduler step.
- VLM is requested whenever an agent's own viewpoint changes, but agents whose viewpoint did not change do not request VLM.

### 6. VLM and Semantic Map Topic Wiring

Checked files:

- `habitat_evaluation.py`
- `src/planner/plan_env/src/map_ros.cpp`
- `src/planner/plan_env/msg/MultipleMasksWithConfidence.msg`

Result: **PASS, static**

Python publishes:

| Topic | Message | Purpose |
|---|---|---|
| `/blip2/agent_X/cosine_score` | `Float64` | ITM semantic relevance score |
| `/detector/agent_X/clouds_with_scores` | `MultipleMasksWithConfidence` | Object point clouds, confidences, labels |

C++ subscribes:

```text
/blip2/agent_X/cosine_score
/detector/agent_X/clouds_with_scores
```

Custom message:

```text
sensor_msgs/PointCloud2[] point_clouds
float32[] confidence_scores
int32[] label_indices
```

Runtime risk: this static check does not verify point cloud quality, mask correctness, detection latency, or VLM server availability.

## Runtime Checks Still Required

These cannot be completed from a non-running workspace. Run them with `roscore`, `exploration.launch`, VLM servers, and `habitat_evaluation.py` active.

### Required Live Commands

FSM state:

```bash
rostopic echo /ros/state_all
rostopic echo /habitat/state
```

Action topics:

```bash
rostopic echo /agent_0/action
rostopic echo /agent_1/action
rostopic echo /agent_2/action
```

Sensor rates:

```bash
rostopic hz /habitat/agent_0/camera_depth
rostopic hz /habitat/agent_0/camera_rgb
rostopic hz /habitat/agent_0/odom
```

Perception and map:

```bash
rostopic hz /detector/agent_0/clouds_with_scores
rostopic hz /blip2/agent_0/cosine_score
rostopic echo /detector/confidence_threshold
rostopic hz /grid_map/occupancy_object
rostopic hz /grid_map/value_map
```

### Required Live Reliability Criteria

Run 10 episodes first. If clean, run 50, then 100.

Pass criteria:

- No permanent `WAIT_ACTION_FINISH`.
- No repeated `resetMap()` spam.
- No premature trigger before all agents reach `WAIT_TRIGGER`.
- Every C++ action receives a matching Python `ACTION_FINISH`.
- Every episode publishes one effective `EPISODE_FINISH`.
- Next episode reaches `WAIT_TRIGGER` without manual restart.
- No TF out-of-order warnings during normal operation.
- VLM requests happen only for agents whose viewpoint changed.
- Object clouds do not leak across episode resets.

Recommended counters to record:

| Counter | Expected |
|---|---|
| episode count | reaches requested total |
| deadlock count | `0` |
| `WAIT_ACTION_FINISH` timeout count | ideally `0`; investigate any repeated burst |
| reset count per episode | `1` effective reset |
| trigger count per episode | `1` intentional trigger |
| VLM server errors | `0` |
| stale object cloud after reset | `0` |

## Current Assessment

| Layer | Static Result | Runtime Confidence |
|---|---|---|
| Python syntax | PASS | High |
| State enum consistency | PASS | High |
| Topic naming consistency | PASS | Medium-high |
| Startup trigger ordering | PASS, code present | Needs live confirmation |
| Episode reset handshake | PASS, code present | Needs live multi-episode confirmation |
| VLM throttling logic | PASS, code present | Needs log confirmation |
| Map reset cleanliness | Not fully testable offline | Needs live RViz/topic confirmation |
| Navigation algorithm quality | Not evaluated offline | Needs benchmark episodes |

## Recommended Next Step

Add a lightweight CSV or JSONL diagnostics logger in `habitat_evaluation.py` before large algorithm work. Minimum useful fields:

```text
episode_id, agent_id, ros_state, habitat_state, action,
count_steps, viewpoint_steps_since_perception, vlm_called,
distance_to_goal, success, object_cloud_count,
action_finish_sent, episode_finish_sent
```

Then run:

1. 10 fixed episodes for handshake validation.
2. 50 fixed episodes for reliability.
3. 100 episodes for regression baseline.

Only after those pass should algorithm changes be compared using SPL, SoftSPL, success rate, false positives, stuck count, and VLM calls per episode.
