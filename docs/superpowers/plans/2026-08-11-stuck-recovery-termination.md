# Multi-Agent Stuck Recovery and Termination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make local collision recovery replan instead of terminating on generic stationary actions, and make team termination/reset explicit, per-agent, and observable.

**Architecture:** Keep the shared map and independent agent FSMs, but replace the generic stationary-action terminal counter with action-aware failed-forward recovery. Publish per-agent final results across the ROS boundary and have Python report an explicit team termination reason before resetting maps.

**Tech Stack:** C++17, ROS1 publishers/subscribers, Python Habitat evaluation, pytest, catkin.

## Global Constraints

- A local stuck event must never broadcast STOP or reset shared maps.
- Rotations and camera actions must not count as failed forward motion.
- Escape exhaustion must replan and must not mark the robot's current cell occupied.
- Existing ROS numeric values and the legacy `/ros/expl_state` topic remain compatible.
- Habitat Matrix output and record writing happen before `EPISODE_FINISH` resets RViz maps.

---

### Task 1: Make FSM terminal-state logging memory safe

**Files:**
- Modify: `src/planner/exploration_manager/include/exploration_manager/exploration_data.h`
- Modify: `src/planner/exploration_manager/src/exploration_fsm.cpp`
- Test: `tests/test_episode_reset_lifecycle.py`

**Interfaces:**
- Produces: a complete state-name mapping and bounds-safe `stateName(ROS_STATE)` logging helper.

- [ ] Add a failing regression test that exercises or inspects the state mapping and rejects a missing `FINISH_FAILURE` name.
- [ ] Run the test and confirm it fails on the six-element mapping.
- [ ] Add `FINISH_FAILURE` and use a bounds-safe state-name helper in `transitState`.
- [ ] Run the targeted test and confirm it passes.

### Task 2: Replace generic stationary termination with failed-forward recovery

**Files:**
- Modify: `src/planner/exploration_manager/include/exploration_manager/exploration_data.h`
- Modify: `src/planner/exploration_manager/include/exploration_manager/exploration_fsm.h`
- Modify: `src/planner/exploration_manager/src/exploration_fsm.cpp`
- Create: `tests/test_stuck_recovery_policy.py`

**Interfaces:**
- Consumes: `ACTION::MOVE_FORWARD`, `STUCKING_DISTANCE`, per-agent odometry and escape state.
- Produces: `registerActionOutcome(...)`-equivalent action-aware accounting and a recovery-exhaustion path that sets `replan_flag_` without returning `STUCKING`.

- [ ] Add failing tests for: rotations do not accumulate stuck evidence; failed forward activates recovery; exhausted recovery does not mark `current_pos`; exhausted recovery forces replan instead of terminal `STUCKING`.
- [ ] Run the new tests and confirm the old generic counter fails them.
- [ ] Implement the minimum action-aware accounting and recovery-state resets.
- [ ] Remove the `MAX_STUCKING_COUNT` terminal branch and current-cell occupancy mutation.
- [ ] Run the new and existing FSM policy tests.

### Task 3: Publish and consume per-agent final results

**Files:**
- Modify: `src/planner/exploration_manager/include/exploration_manager/exploration_fsm.h`
- Modify: `src/planner/exploration_manager/src/exploration_fsm.cpp`
- Modify: `habitat_evaluation.py`
- Modify: `tests/test_multiagent_success_policy.py`

**Interfaces:**
- Produces: `/ros/final_result_all` (`std_msgs/Int32MultiArray`) indexed by agent.
- Consumes: the same array in Python as `ros_final_results`.

- [ ] Add failing tests that require the new topic and require multi-agent failure classification to use the report agent's result.
- [ ] Run tests and confirm failure because only scalar `/ros/expl_state` exists.
- [ ] Publish the array with planner state updates and subscribe in Python.
- [ ] Use the report agent's final result when building the Matrix failure classification.
- [ ] Run the targeted policy tests.

### Task 4: Make team termination and reset ordering explicit

**Files:**
- Modify: `basic_utils/failure_check/failure_check.py`
- Modify: `habitat_evaluation.py`
- Modify: `tests/test_multiagent_success_policy.py`
- Modify: `tests/test_episode_reset_lifecycle.py`

**Interfaces:**
- Produces: `get_multiagent_termination_reason(...) -> Optional[str]` with literal results `reach_claim`, `all_failed`, and `step_limit`.
- Consumes: the reason in the Habitat loop and episode-end report.

- [ ] Add failing table-driven tests for every termination reason and a regression test that requires result output before `_finish_episode_handshake()`.
- [ ] Run tests and confirm failure because termination is currently only boolean and reset precedes reporting.
- [ ] Implement the reason helper and log a per-agent termination snapshot.
- [ ] Move the reset handshake to after Matrix/record finalization while preserving reset-before-next-episode behavior.
- [ ] Run all termination and lifecycle tests.

### Task 5: Verify the integrated change

**Files:**
- Test: `tests/test_stuck_recovery_policy.py`
- Test: `tests/test_episode_reset_lifecycle.py`
- Test: `tests/test_multiagent_success_policy.py`

- [ ] Run targeted pytest tests.
- [ ] Run `python -m py_compile habitat_evaluation.py basic_utils/failure_check/failure_check.py params.py`.
- [ ] Build the affected catkin package if ROS dependencies are available.
- [ ] Run `git diff --check` and inspect that unrelated dirty-worktree changes were preserved.
