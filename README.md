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
| `agent_2` | `1.7 m` | `1.7 m` | `1.7 m` |

The evaluation success distance is:

```text
0.5 m
```

Config key:

```yaml
success_distance: 0.5
```

The maximum episode length for evaluation is:

```yaml
max_episode_steps: 500
```

Step counting policy:

```text
Every executed Habitat action increments the per-agent step counter.
```

This matches the original single-agent counting semantics: forward motion,
turning, looking up/down, and stop actions all count as one step for the agent
that executed the action.

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
