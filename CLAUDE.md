# ApexNav Architectural Notes

## Multi-Agent Architecture: C++ Planner

Single C++ planner node controls all configured simulation agents (`agent_0`, `agent_1`, `agent_2`) within one process.

- `ExplorationFSM` (`exploration_fsm.cpp`) allocates per-agent subscribers and publishers in loops:
  - Odom subscribers: `/habitat/agent_{i}/odom`
  - Action publishers: `/habitat/plan_action_agent_{i}`
  - Robot marker publishers: `/robot_agent_{i}`
  The FSM loop iterates `for (int agent_idx = 0; agent_idx < NUM_AGENTS; ++agent_idx)` to plan sequentially.
  Do not launch multiple planner instances; that creates duplicate publishers, separate maps, and race conditions.

- The map building module (`plan_env/map_ros.cpp`) dynamically subscribes per agent:
  - Reads `sensor_pose_topic` and `depth_topic` params from `algorithm.xml`, using `agent_0` topics as templates.
  - Uses `makeAgentTopic()` to replace only the digit sequence in `agent_X`, preserving the suffix.
  - Subscribes `/detector/agent_X/clouds_with_scores` and `/blip2/agent_X/cosine_score` per agent.
  - All agents contribute to one shared SDF/Object/ValueMap via `map_mutex_`.

- `NUM_AGENTS = 3` is a compile-time constant in `exploration_data.h`.
- `MapROS::NUM_AGENTS_ = 3` is a compile-time constant in `map_ros.h`.

## Data Flow (Multi-Agent)

```text
habitat_evaluation.py (MultiAgentSim-v0, 3 agents)
  pub: /habitat/agent_0/{camera_rgb, camera_depth, sensor_pose, odom}
       /habitat/agent_1/{camera_rgb, camera_depth, sensor_pose, odom}
       /habitat/agent_2/{camera_rgb, camera_depth, sensor_pose, odom}
       /habitat/state
       /move_base_simple/goal
  sub: /habitat/plan_action_agent_0
       /habitat/plan_action_agent_1
       /habitat/plan_action_agent_2

exploration.launch -> C++ planner (exploration_node, 1 instance, shared map)
  MapROS per-agent subs:
    depth+pose: /habitat/agent_X/{camera_depth, sensor_pose}
    object:     /detector/agent_X/clouds_with_scores
    ITM:        /blip2/agent_X/cosine_score
  ExplorationFSM per-agent subs:
    odom:       /habitat/agent_X/odom
  per-agent pubs:
    action:     /habitat/plan_action_agent_X
```

## Running Simulation

1. `roslaunch exploration_manager exploration.launch`
2. `python habitat_evaluation.py --dataset hm3dv2`

Both default to the HM3D-v2 config path. The active simulation config is `config/habitat_eval_hm3dv2.yaml` with `num_agents: 3`.

## `real_world_test_habitat.py` Is Not Needed For Sim Mode

`habitat_evaluation.py` already does perception in the simulation loop. Running `real_world_test_habitat.py` alongside it causes duplicate publishers on `/detector/agent_X/clouds_with_scores` and `/blip2/agent_X/cosine_score`.

## Config Files

- `config/habitat_eval_hm3dv2.yaml` - triple-agent simulation config.
- `config/habitat_eval_hm3dv1.yaml` - triple-agent simulation config.
- `config/habitat_eval_mp3d.yaml` - triple-agent simulation config.
- `config/habitat_vel_control.yaml` - separate velocity-control pipeline; `habitat_vel_control.py` remains single-agent.

## Important Gotchas

- Action encoding for multi-agent messages can be `action = agent_idx * 100 + action_code`, but the current per-topic planner path sends raw action codes on `/habitat/plan_action_agent_{i}`.
- Episode termination policies: `cooperative` means any agent can end the episode; `independent` means all agents must finish.
- `algorithm.xml` sets `sensor_pose_topic=/habitat/agent_0/sensor_pose` and `depth_topic=/habitat/agent_0/camera_depth`. `MapROS::makeAgentTopic()` derives `agent_1` and `agent_2` topics from those templates.
- The `/odom_world` remap in `algorithm.xml` is not used by the simulation FSM; odom subscribers are created directly from `/habitat/agent_{i}/odom`.
