# States and Workflow

This note summarizes the Python-Habitat state, C++ planner FSM state, action values, and the normal handshake between `habitat_evaluation.py` and the exploration planner.

## State Definitions

### Python / Habitat State

Python publishes Habitat execution state on `/habitat/state`.

Defined in:

- `params.py`
- `src/planner/exploration_manager/include/exploration_manager/exploration_fsm.h`

```python
class HABITAT_STATE:
    READY = 0
    ACTION_EXEC = 1
    ACTION_FINISH = 2
    EPISODE_FINISH = 3
```

| Value | Name | Publisher | Meaning |
|---:|---|---|---|
| `0` | `READY` | Python, mostly legacy | Habitat/Python ready |
| `1` | `ACTION_EXEC` | Python | Python is executing an action in Habitat |
| `2` | `ACTION_FINISH` | Python | Habitat action finished; C++ may plan the next action |
| `3` | `EPISODE_FINISH` | Python | Episode ended; C++ should reset FSM/maps |

### C++ Planner FSM State

C++ publishes per-agent FSM state on `/ros/state_all`.

Defined in:

- `params.py`
- `src/planner/exploration_manager/include/exploration_manager/exploration_fsm.h`

```python
class ROS_STATE:
    INIT = 0
    WAIT_TRIGGER = 1
    PLAN_ACTION = 2
    WAIT_ACTION_FINISH = 3
    PUB_ACTION = 4
    FINISH = 5
```

| Value | Name | Owner | Meaning |
|---:|---|---|---|
| `0` | `INIT` | C++ | Waiting for odom and confidence threshold |
| `1` | `WAIT_TRIGGER` | C++ | Ready, waiting for Python trigger |
| `2` | `PLAN_ACTION` | C++ | Planner chooses the next action |
| `3` | `WAIT_ACTION_FINISH` | C++ | Waiting for Python `/habitat/state = ACTION_FINISH` |
| `4` | `PUB_ACTION` | C++ | Publish selected action to Python |
| `5` | `FINISH` | C++ | Agent or episode complete |

### Action Values

Defined in `params.py`.

```python
class ACTION:
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    TURN_DOWN = 4
    TURN_UP = 5
```

| Value | Name | Meaning |
|---:|---|---|
| `0` | `STOP` | Stop agent |
| `1` | `MOVE_FORWARD` | Move forward |
| `2` | `TURN_LEFT` | Turn left |
| `3` | `TURN_RIGHT` | Turn right |
| `4` | `TURN_DOWN` | Look down |
| `5` | `TURN_UP` | Look up |

Example log:

```text
WAIT_ACTION_FINISH timeout, re-publishing action 4
```

means C++ is waiting for Python to finish `TURN_DOWN`.

## C++ FSM Workflow

Main FSM loop:

- `src/planner/exploration_manager/src/exploration_fsm.cpp`

Normal startup and action cycle:

```text
INIT
  waits for odom + confidence threshold
  |
  v
WAIT_TRIGGER
  waits for /move_base_simple/goal
  |
  v
PLAN_ACTION
  decides next action
  |
  v
PUB_ACTION
  publishes /agent_X/action
  |
  v
WAIT_ACTION_FINISH
  waits for /habitat/state = ACTION_FINISH
  |
  v
PLAN_ACTION
  repeats
```

Episode finish/reset:

```text
Python publishes /habitat/state = EPISODE_FINISH
  |
  v
C++ habitatStateCallback()
  |
  v
resetEpisode()
  |
  v
state_[all agents] = INIT
  |
  v
maps/frontiers/object map/value map reset
  |
  v
wait for fresh odom + confidence
  |
  v
WAIT_TRIGGER
```

Important C++ locations:

- `ExplorationFSM::execFSMCallback()`: main state machine
- `ExplorationFSM::habitatStateCallback()`: receives `ACTION_FINISH` and `EPISODE_FINISH`
- `ExplorationFSM::triggerCallback()`: receives `/move_base_simple/goal`
- `ExplorationFSM::resetEpisode()`: resets FSM state and maps between episodes

## Python Workflow

Main Python loop:

- `habitat_evaluation.py`

Per episode:

```text
env.reset()
  |
  v
publish RGB/depth/odom/confidence
  |
  v
wait until all C++ agents report WAIT_TRIGGER
  |
  v
publish /move_base_simple/goal once
  |
  v
main action loop
```

Main action loop:

```text
Python waits for C++ action topic
  |
  v
publish /habitat/state = ACTION_EXEC
  |
  v
env.step(action)
  |
  v
publish latest observations/odom
  |
  v
maybe run VLM if viewpoint changed
  |
  v
publish /habitat/state = ACTION_FINISH
  |
  v
C++ leaves WAIT_ACTION_FINISH and plans next action
```

Episode end:

```text
episode done / success / max steps / finish condition
  |
  v
Python calls _finish_episode_handshake()
  |
  v
publish /habitat/state = EPISODE_FINISH
  |
  v
wait for C++ state INIT or WAIT_TRIGGER
  |
  v
record metrics
  |
  v
switch env.current_episode
  |
  v
next episode
```

Important Python locations:

- ROS setup and state subscriptions: `habitat_evaluation.py`
- Wait for C++ `WAIT_TRIGGER`: readiness loop before the main action loop
- Publish trigger: `trigger_pub.publish(PoseStamped())`
- Execute action and publish `ACTION_EXEC`
- Publish `ACTION_FINISH`
- Episode finish handshake: `_finish_episode_handshake()`

## Python-C++ Handshake Summary

```text
C++ WAIT_TRIGGER
  <- Python publishes /move_base_simple/goal

C++ PUB_ACTION
  -> publishes ACTION

Python receives ACTION
  -> publishes /habitat/state = ACTION_EXEC
  -> env.step(action)
  -> publishes /habitat/state = ACTION_FINISH

C++ WAIT_ACTION_FINISH
  <- receives ACTION_FINISH
  -> goes PLAN_ACTION again

Episode end:
Python publishes /habitat/state = EPISODE_FINISH
  -> C++ resetEpisode()
  -> C++ INIT
  -> C++ WAIT_TRIGGER after odom/confidence
```

## Common Deadlock Pattern

If Python prints:

```text
Waiting for trigger: [agent_0: state=3, agent_1: state=3, agent_2: state=3]
```

and C++ prints:

```text
WAIT_ACTION_FINISH timeout, re-publishing action 4
```

then C++ has already entered `WAIT_ACTION_FINISH`, but Python is still waiting for `WAIT_TRIGGER`. This usually means C++ was triggered too early or Python missed the action-finish handshake.

The readiness timer should publish odom/depth/confidence only. It must not publish `/move_base_simple/goal` until all agents have reached `WAIT_TRIGGER`.
