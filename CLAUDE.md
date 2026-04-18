# ApexNav — Architectural Notes

## Multi-Agent Architecture: C++ Planner

**Single C++ planner node controls BOTH agents (`agent_0` and `agent_1`)** within one process.

- `ExplorationFSM` (`exploration_fsm.cpp`) hardcodes odom subscribers and per-agent action publishers:
  ```cpp
  odom_sub_[0] = subscribe("/habitat/agent_0/odom", 10, odometryCallback0, this);
  odom_sub_[1] = subscribe("/habitat/agent_1/odom", 10, odometryCallback1, this);
  action_pub_[0] = advertise("/habitat/plan_action_agent_0", 10);
  action_pub_[1] = advertise("/habitat/plan_action_agent_1", 10);
  ```
  The FSM loop iterates `for (int agent_idx = 0; agent_idx < NUM_AGENTS; ++agent_idx)` to plan sequentially.
  **Do NOT launch two planner instances** — this creates duplicate publishers, separate maps, and race conditions.

- The map building module (`plan_env/map_ros.cpp`) dynamically subscribes per-agent:
  - Reads `sensor_pose_topic` and `depth_topic` params from `algorithm.xml` (e.g. `/habitat/agent_0/camera_depth`)
  - Uses `makeAgentTopic()` to replace only the digit in `agent_X`, preserving the suffix (e.g. `/habitat/agent_0/camera_depth` → `/habitat/agent_1/camera_depth`)
  - Subscribes `/detector/agent_X/clouds_with_scores` and `/blip2/agent_X/cosine_score` per agent
  - **All agents contribute to one shared SDF/Object/ValueMap via `map_mutex_`**

- `NUM_AGENTS = 2` is a compile-time constant in `exploration_data.h`

## Data Flow (Multi-Agent)

```
habitat_evaluation.py (MultiAgentEnv, 2 agents)
  pub: /habitat/agent_0/{camera_rgb, camera_depth, sensor_pose, odom}
       /habitat/agent_1/{camera_rgb, camera_depth, sensor_pose, odom}
       /habitat/state (→ C++ planner FSM state)
       /move_base_simple/goal (→ C++ planner trigger)
  sub: /habitat/plan_action_agent_0 (from C++ planner)
       /habitat/plan_action_agent_1 (from C++ planner)

exploration.launch → C++ planner (exploration_node, 1 instance, manages both agents)
  MapROS per-agent subs:
    map_ros depth+pose: /habitat/agent_0/{camera_depth, sensor_pose}  (via params + makeAgentTopic for agent_1)
    object cloud:       /detector/agent_0/clouds_with_scores
    ITM score:          /blip2/agent_0/cosine_score
    (same for agent_1)
  ExplorationFSM per-agent subs:
    odom:               /habitat/agent_0/odom, /habitat/agent_1/odom  (hardcoded topics)
  per-agent pubs:
    action:             /habitat/plan_action_agent_0, /habitat/plan_action_agent_1

real_world_test_habitat.py (perception-only, multi-agent ROS consumer)
  sub: /habitat/agent_0/{camera_rgb, camera_depth, sensor_pose}
       /habitat/agent_1/{camera_rgb, camera_depth, sensor_pose}
  pub: /detector/agent_X/clouds_with_scores  (SAME TOPICS as C++ planner subscribes!)
       /blip2/agent_X/cosine_score           (SAME TOPICS as C++ planner subscribes!)
```

## Key Topic Naming Convention

All agent-namespaced topics use `/habitat/agent_{i}/` prefix. The `habitat2ros/habitat_publisher.py` handles this via `ROSPublisher(agent_name)` constructor.

## Running Multi-Agent

1. `roslaunch exploration_manager exploration.launch` (single C++ node, internal multi-agent, shared map)
2. `python habitat_evaluation.py --dataset hm3dv2` (num_agents=2 from config)

## `real_world_test_habitat.py` is NOT needed for sim mode

`habitat_evaluation.py` multi-agent loop already does perception (ITM + detection) internally. Running `real_world_test_habitat.py` alongside **causes duplicate publishers** on `/detector/agent_X/clouds_with_scores` and `/blip2/agent_X/cosine_score` — the C++ planner subscribes to both, so both `habitat_evaluation.py` AND `real_world_test_habitat.py` publishing to the same topics creates conflicts.

`real_world_test_habitat.py` is designed for `habitat_vel_control.py` (single-agent velocity control), which does NOT do perception. It's a separate pipeline.

## Config Files

- `config/habitat_eval_hm3dv2.yaml` — multi-agent ready (num_agents: 2, agent_1 defined, multiagent section)
- `config/habitat_eval_hm3dv1.yaml` — multi-agent ready
- `config/habitat_eval_mp3d.yaml` — multi-agent ready
- `config/habitat_vel_control.yaml` — multi-agent ready for sim side
  **but habitat_vel_control.py itself is single-agent**

## Important Gotchas

- `habitat_vel_control.py`  is **single-agent only** (separate velocity control experiment, not part of multi-agent)
- Action encoding for multi-agent: `action = agent_idx * 100 + action_code`, single-agent uses raw code (`< 10`)
- Episode termination policies: `cooperative` (any agent finishes → episode ends) or `independent` (all must finish)
- `map_ros.cpp`'s `makeAgentTopic()` derives per-agent topics by replacing only the digit after `agent_` in the base topic string — preserves suffix like `/camera_depth`. Only works if the param value contains `agent_X`
- `algorithm.xml` sets params `sensor_pose_topic=/habitat/agent_0/sensor_pose` and `depth_topic=/habitat/agent_0/camera_depth`. MapROS reads these and `makeAgentTopic` replaces `0` with `1` for agent_1. The old remap lines are still present but unused by MapROS.
- The `/odom_world` remap in `algorithm.xml` is unused by the C++ planner — odom topics are hardcoded
