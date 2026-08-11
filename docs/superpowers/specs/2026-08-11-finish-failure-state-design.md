# Per-Agent Failure Terminal State Design

## Goal

Separate a planner agent's unsuccessful exit from the team-level successful
terminal state. A single failed agent must stop acting without ending a
cooperative episode; the episode fails only when every agent has failed (or
the Habitat hard step limit is exhausted).

## State protocol

- Preserve all existing ROS state numeric values.
- Append `FINISH_FAILURE = 6` in both the C++ and Python state definitions.
- `STUCKING` and `NO_FRONTIER` enter `FINISH_FAILURE`.
- `REACH_OBJECT` keeps the existing atomic reach-claim path and moves all
  agents to `FINISH` after broadcasting STOP.
- A failed agent publishes one local STOP, then remains inactive. It never
  broadcasts STOP to another agent.

## Python termination behavior

- Python reads `/ros/state_all` as the per-agent terminal-state authority.
- One `FINISH_FAILURE` marks only that agent inactive; remaining agents keep
  exploring against the shared map.
- All agents in `FINISH_FAILURE` end the episode as an overall failure.
- A later valid reach claim from an active agent takes precedence and follows
  the existing Success/False Positive evaluation path.
- `max_episode_steps` remains an independent hard safety limit.

## Reset and visualization

Neither local failure nor entry into `FINISH_FAILURE` resets shared maps.
Maps are cleared only by the existing Python `EPISODE_FINISH` handshake after
the complete episode terminates. Idle callbacks accept both terminal states.

## Verification

- Protocol tests assert matching state numbers in C++ and Python.
- Policy tests assert that one failure does not terminate and all failures do.
- C++ tests assert failure results transition to `FINISH_FAILURE` while
  `REACH_OBJECT` still transitions to `FINISH`.
- Build `exploration_manager` and run the targeted Python policy tests.
