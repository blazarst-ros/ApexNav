# Multi-Agent Stuck Recovery and Termination Design

## Goal

Prevent a locally stuck robot from corrupting the shared FSM or prematurely
ending cooperative exploration. A failed forward move must trigger recovery
and replanning; shared maps may be reset only after an explicit team-level
episode termination.

## Confirmed failure chain

- Agent 0 reached the old generic stationary-action threshold and returned
  `FINAL_RESULT::STUCKING`.
- The FSM transitioned agent 0 to `ROS_STATE::FINISH_FAILURE`; agent 1 remained
  active until it exhausted the Habitat step budget.
- Python then published `HABITAT_STATE::EPISODE_FINISH`, and only that message
  called `resetEpisode()` and cleared RViz maps.
- `FSMData::state_str_` omitted the new `FINISH_FAILURE` entry. Logging the
  transition indexed beyond the vector, producing undefined behavior and the
  corrupted transition line in the runtime log.
- The old stuck counter treated rotations and camera actions as failed movement,
  while recovery activation considered only `MOVE_FORWARD`. This mismatch could
  terminate an agent that was still executing a valid turn sequence.

## C++ FSM behavior

- State names cover every `ROS_STATE` value and transition logging uses a
  bounds-safe state-name helper.
- Stuck evidence is derived only from a completed `MOVE_FORWARD` whose planar
  displacement is below `STUCKING_DISTANCE`.
- Turning and camera actions neither increment the failed-forward counter nor
  cause terminal failure.
- A failed forward starts the existing escape sequence. If the sequence moves
  the robot, recovery state and counters are cleared.
- If the escape sequence is exhausted, mark only the attempted forward cells as
  occupied. Never mark the robot's current cell occupied. Release the current
  frontier claim, force replanning, and continue the agent FSM.
- Repeated failure at the same pose also forces obstacle marking and replanning;
  it does not return `STUCKING` solely because the robot was stationary.
- An agent enters `FINISH_FAILURE` only when replanning returns no viable path or
  no viable exploration target. A local failure still never resets shared maps.

## Per-agent result protocol

- Publish `/ros/final_result_all` as an `Int32MultiArray`, indexed by agent.
- Python subscribes to this array and uses the selected/reporting agent's final
  result for failure classification. The legacy scalar `/ros/expl_state` remains
  available for compatibility but is not authoritative in multi-agent mode.
- Terminal strategy output is updated for failed agents instead of repeatedly
  publishing stale frontier/object targets.

## Episode termination and reset

- Python computes an explicit termination reason: reach claim completed, all
  agents failed, or every non-failed agent exhausted its step budget.
- Before publishing `EPISODE_FINISH`, Python logs the reason, per-agent ROS
  states, final results, steps, and finished flags.
- Metrics and the Habitat Matrix are finalized before the reset handshake.
- `resetEpisode()` remains reachable only from an explicit
  `HABITAT_STATE::EPISODE_FINISH` message.

## Observability

Record action commands, FSM states, per-agent planner results, odometry, strategy,
the Habitat state handshake, reach claims, transforms, and principal grid-map
topics. These signals distinguish physical collision, planner failure, Python
step-budget termination, and the subsequent map reset.

## Verification

- Tests cover complete state-name mapping and bounds-safe transition logging.
- Tests cover action-aware stuck accounting and recovery exhaustion replanning.
- Tests cover per-agent final-result publication/subscription.
- Tests cover explicit termination reasons and Matrix-before-reset ordering.
- Run targeted pytest tests, Python compilation, a catkin build when the local
  ROS environment is available, and `git diff --check`.
