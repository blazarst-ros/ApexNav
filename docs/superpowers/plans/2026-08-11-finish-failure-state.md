# Per-Agent Failure Terminal State Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a distinct per-agent failure terminal state so cooperative exploration ends only after all agents fail or a team-level terminal condition occurs.

**Architecture:** Append a wire-compatible `FINISH_FAILURE` ROS state in C++ and Python. Route planner failure results into that state, treat it as locally inactive, and make Python require all agents to report it before declaring cooperative failure while retaining Habitat's hard step limit.

**Tech Stack:** C++17 ROS FSM, Python Habitat evaluation, pytest-style policy tests, catkin.

## Global Constraints

- Preserve existing ROS state numeric values 0 through 5.
- `STUCKING` and `NO_FRONTIER` are failure terminal results.
- `REACH_OBJECT` retains the current reach-claim and global STOP behavior.
- A single failed agent must not reset shared maps or terminate active peers.

---

### Task 1: Lock the terminal-state protocol with failing tests

**Files:**
- Modify: `tests/test_multiagent_success_policy.py`
- Modify: `tests/test_episode_reset_lifecycle.py`

- [ ] Add tests for matching `FINISH_FAILURE = 6`, C++ failure routing, local failure idling, all-failed Python termination, and unchanged reach success routing.
- [ ] Run the targeted tests and confirm they fail because the new state and policy do not exist.

### Task 2: Implement C++ failure terminal behavior

**Files:**
- Modify: `src/planner/exploration_manager/include/exploration_manager/exploration_fsm.h`
- Modify: `src/planner/exploration_manager/src/exploration_fsm.cpp`

- [ ] Append `FINISH_FAILURE` without renumbering existing states.
- [ ] Add a local failure terminal branch that publishes STOP once and logs the stored failure reason.
- [ ] Route all non-success terminal planner results to `FINISH_FAILURE`.
- [ ] Treat `FINISH_FAILURE` as idle in the frontier callback.

### Task 3: Implement Python cooperative failure handling

**Files:**
- Modify: `params.py`
- Modify: `habitat_evaluation.py`

- [ ] Mirror `FINISH_FAILURE = 6` in Python.
- [ ] Mark failed agents inactive without conflating them with success FINISH.
- [ ] Break on all failed agents, while retaining the all-inactive/max-step safety condition.
- [ ] Keep the reach-claim STOP execution gate unchanged.

### Task 4: Verify integration

**Files:**
- Test: `tests/test_multiagent_success_policy.py`
- Test: `tests/test_episode_reset_lifecycle.py`

- [ ] Run targeted policy tests and all directly runnable local tests.
- [ ] Compile Python sources.
- [ ] Build `exploration_manager` with catkin.
- [ ] Run `git diff --check` and inspect the final diff.
