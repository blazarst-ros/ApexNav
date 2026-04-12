python habitat_evaluation.py --dataset hm3dv2
    │
    ├── 1. Load YAML config → type: MultiAgentSim-v0, agents: {agent_0, agent_1}
    │
    ├── 2. patch_config(cfg) → sets agents_order = [agent_0, agent_1]
    │
    ├── 3. habitat.Env(cfg)
    │     ├── make_sim(type="MultiAgentSim-v0", config=cfg.simulator)
    │     │     ├── MultiAgentSim-v0.__init__(config)
    │     │     │   ├── Rename sensor UUIDs: agent_0_rgb, agent_1_rgb, etc.
    │     │     │   ├── HabitatSim.__init__() → create_sim_config()
    │     │     │   │   └── Build [AgentConfig_0, AgentConfig_1]
    │     │     │   │       → habitat_sim.Configuration(sim_cfg, [cfg_0, cfg_1])
    │     │     │   │       → habitat-sim C++ creates 2 Agent objects
    │     │     │   └── _action_space, _sensor_suite set up
    │     │     └── Simulator object created with 2 agents inside
    │     │
    │     └── make_task(type="ObjectNav-v1") → Task with single-agent measurements
    │
    ├── 4. env.reset()
    │     ├── task.reset(episode) → sim.reset()
    │     │   ├── habitat_sim.Simulator.reset() → repositions agent_0 to episode start
    │     │   ├── _update_agents_state() → sets start positions for all agents
    │     │   └── get_sensor_observations(agent_ids=[0,1])
    │     │       → {0: {agent_0_rgb: ..., agent_0_depth: ...},
    │     │          1: {agent_1_rgb: ..., agent_1_depth: ...}}
    │     └── _get_all_agent_observations() → flatten + SensorSuite →
    │         {agent_0_rgb, agent_0_depth, agent_1_rgb, agent_1_depth, gps, compass, ...}
    │
    ├── 5. _get_agent_observations() → split by prefix:
    │     {agent_0: {rgb, depth, gps, compass}, agent_1: {rgb, depth, gps, compass}}
    │
    ├── 6. _setup_multi_agent_env() → offset agent_1 spawn position
    │     env.sim.set_agent_state(pos, rot, agent_id=1) 
    │
    ├── 7. ROS publish per-agent observations via ros_pubs[agent_name]
    │     /habitat/agent_0/camera_depth, /habitat/agent_0/camera_rgb, ...
    │     /habitat/agent_1/camera_depth, /habitat/agent_1/camera_rgb, ...
    │
    └── 8. Episode loop: receive actions from C++ planner
          ├── _parse_multi_agent_action(raw) → (agent_idx, action_code)
          ├── _multi_agent_step(env, {agent_0: forward, agent_1: left})
          │     ├── env.sim.get_agent(0).act(ActionSpec("move_forward"))
          │     ├── env.sim.get_agent(1).act(ActionSpec("turn_left"))
          │     ├── env.sim.step_world(1/60)
          │     └── _get_agent_observations()
          └── publish per-agent ITM scores + object point clouds
                /blip2/agent_0/cosine_score  /detector/agent_0/clouds_with_scores
                /blip2/agent_1/cosine_score  /detector/agent_1/clouds_with_scores