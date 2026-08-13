# 按模式门控的双 Agent MTSP 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 在 ApexNav 中实现模式相同时集中双起点 `MINMAX` mTSP 动态规划、模式不同时独立规划的 A* 距离分配。

**架构：** FSM 在同一规划周期先评估两台 Agent 的最高优先级模式。模式不同或只有一台需要重规划时保留无 Claim 的独立规划；两台均需重规划且模式一致时，对最多 10 个同模式候选运行 Held--Karp 双起点 `MINMAX` 动态规划。动态规划回溯两条路线，FSM 只下发每条路线的首目标。

**技术栈：** C++14、ROS1、Eigen、Astar2D、Held--Karp 动态规划、现有 LKH-3 单 Agent ATSP、pytest、catkin。

## 全局约束

- 固定 `NUM_AGENTS == 2`。
- 联合分配不调用当前单 depot LKH-MTSP；使用两个不同起点的 Held--Karp `MINMAX` 动态规划。
- 代价必须来自 A* 路径长度，语义值和置信度只作候选门控。
- `SEARCH_BEST_OBJECT` 仅使用高置信度对象；不混入其他对象或 Frontier。
- 常规规划不读写 `Frontier2D::claimed_by_`。
- 求解、解析、路径校验失败时安全回退独立规划。
- 仅下发各自路线的第一个目标，继续使用现有滚动重规划。

---

### 任务 1：定义模式与联合分配契约

**文件：**

- 修改：`src/planner/exploration_manager/include/exploration_manager/exploration_manager.h`
- 修改：`src/planner/exploration_manager/include/exploration_manager/exploration_data.h`
- 修改：`src/planner/exploration_manager/src/exploration_manager.cpp`
- 测试：`tests/test_mode_gated_mtsp_policy.py`

**接口：**

- `enum class NavigationMode { SEARCH_BEST_OBJECT, SEARCH_OVER_DEPTH_OBJECT, SEMANTIC_FRONTIER, GEOMETRIC_FRONTIER, SUSPICIOUS_OBJECT, DORMANT_FRONTIER, EXTREME, NONE }`。
- `NavigationMode evaluateNavigationMode(const Vector3d& pos, int agent_idx)`：无副作用地复用现有优先级判定。
- `JointAssignment planJointModeTargets(const std::array<Vector3d, NUM_AGENTS>& positions, NavigationMode mode)`：返回两个首目标、两条路径和有效标志。

- [ ] **步骤 1：写失败测试**

```python
def test_joint_mtsp_contract_exposes_modes_and_two_agent_assignment():
    header = read("src/planner/exploration_manager/include/exploration_manager/exploration_manager.h")
    for token in ("NavigationMode", "SEARCH_BEST_OBJECT", "SEARCH_OVER_DEPTH_OBJECT",
                  "SEMANTIC_FRONTIER", "GEOMETRIC_FRONTIER", "JointAssignment",
                  "evaluateNavigationMode", "planJointModeTargets", "std::array"):
        assert token in header
```

- [ ] **步骤 2：验证失败**

运行：`pytest tests/test_mode_gated_mtsp_policy.py::test_joint_mtsp_contract_exposes_modes_and_two_agent_assignment -v`

预期：因接口不存在而失败。

- [ ] **步骤 3：最小实现**

在 `ExplorationManager` 中声明上述类型和方法。将当前 `planNextBestPoint` 内的优先级判断提取为只读的模式评估入口；不在此任务修改 FSM 调度。

- [ ] **步骤 4：验证通过**

运行：`pytest tests/test_mode_gated_mtsp_policy.py::test_joint_mtsp_contract_exposes_modes_and_two_agent_assignment -v`

预期：通过。

- [ ] **步骤 5：提交**

运行：`git add src/planner/exploration_manager/include/exploration_manager/exploration_manager.h src/planner/exploration_manager/include/exploration_manager/exploration_data.h src/planner/exploration_manager/src/exploration_manager.cpp tests/test_mode_gated_mtsp_policy.py; git commit -m "feat: add two-agent MTSP planning contract"`

### 任务 2：实现双起点 MINMAX 动态规划与路线回溯

**文件：**

- 修改：`src/planner/exploration_manager/src/exploration_manager.cpp`
- 测试：`tests/test_mode_gated_mtsp_policy.py`

**接口：**

- `bool solveTwoStartMinmax(...)`：以两个 Agent 起点、A* 矩阵和候选列表产生两条互斥路线。
- `bool reconstructRoute(...)`：从 DP 前驱表恢复一个 Agent 的目标索引序列。

- [ ] **步骤 1：写失败测试**

```python
def test_two_agent_mtsp_uses_two_start_minmax_dynamic_programming():
    source = read("src/planner/exploration_manager/src/exploration_manager.cpp")
    assert "solveTwoStartMinmax" in source
    assert "held_karp" in source.lower()
    assert "agent_positions[0]" in source
    assert "agent_positions[1]" in source

def test_two_start_dp_validates_duplicate_and_out_of_range_targets():
    source = read("src/planner/exploration_manager/src/exploration_manager.cpp")
    assert "candidate_count" in source
    assert "duplicate" in source.lower()
```

- [ ] **步骤 2：验证失败**

运行：`pytest tests/test_mode_gated_mtsp_policy.py -k "minmax or validates" -v`

预期：因双起点 DP 与路线回溯不存在而失败。

- [ ] **步骤 3：最小实现**

以 `K <= 10` 的候选上限实现 Held--Karp 表：分别计算 Agent 0、Agent 1 从各自起点访问任意子集的最短开放路线。遍历互补子集，选择 `max(route0_cost, route1_cost)` 最小的划分；从前驱表回溯两条路线，并拒绝空路线、重复、遗漏、范围外索引和不可达路径。

- [ ] **步骤 4：验证通过**

运行：`pytest tests/test_mode_gated_mtsp_policy.py -k "minmax or validates" -v`

预期：通过。

- [ ] **步骤 5：提交**

运行：`git add src/planner/exploration_manager/src/exploration_manager.cpp tests/test_mode_gated_mtsp_policy.py; git commit -m "feat: add two-start minmax task solver"`

### 任务 3：构造模式专属共同候选池和 A* 距离矩阵

**文件：**

- 修改：`src/planner/exploration_manager/include/exploration_manager/exploration_manager.h`
- 修改：`src/planner/exploration_manager/src/exploration_manager.cpp`
- 测试：`tests/test_mode_gated_mtsp_policy.py`

**接口：**

- `bool collectJointCandidates(NavigationMode mode, const std::array<Vector3d, NUM_AGENTS>& positions, vector<JointCandidate>& candidates)`。
- `Eigen::MatrixXd computeJointMtspCostMatrix(...)`：显式使用 `R0 -> Ti`、`R1 -> Ti` 与 `Ti -> Tj` 的 A* 代价。

- [ ] **步骤 1：写失败测试**

```python
def test_joint_candidate_pools_are_mode_specific_and_costs_use_both_agent_starts():
    source = read("src/planner/exploration_manager/src/exploration_manager.cpp")
    for token in ("collectJointCandidates", "SEARCH_BEST_OBJECT",
                  "SEARCH_OVER_DEPTH_OBJECT", "SEMANTIC_FRONTIER",
                  "GEOMETRIC_FRONTIER", "computeJointMtspCostMatrix",
                  "agent_positions[0]", "agent_positions[1]"):
        assert token in source
```

- [ ] **步骤 2：验证失败**

运行：`pytest tests/test_mode_gated_mtsp_policy.py::test_joint_candidate_pools_are_mode_specific_and_costs_use_both_agent_starts -v`

预期：因共同候选和双起点矩阵不存在而失败。

- [ ] **步骤 3：最小实现**

分别建立高置信度对象、过深对象、语义 Frontier、几何 Frontier 的候选分支。每个候选必须对两台 Agent 都有有效 A* 路径；目标间距离也由 `computePathCost` 获取。语义分数和置信度只在此处作为准入条件。

- [ ] **步骤 4：验证通过**

运行：`pytest tests/test_mode_gated_mtsp_policy.py::test_joint_candidate_pools_are_mode_specific_and_costs_use_both_agent_starts -v`

预期：通过。

- [ ] **步骤 5：提交**

运行：`git add src/planner/exploration_manager/include/exploration_manager/exploration_manager.h src/planner/exploration_manager/src/exploration_manager.cpp tests/test_mode_gated_mtsp_policy.py; git commit -m "feat: build mode-specific MTSP candidate pools"`

### 任务 4：在 FSM 集中调度并移除常规 Claim

**文件：**

- 修改：`src/planner/exploration_manager/include/exploration_manager/exploration_fsm.h`
- 修改：`src/planner/exploration_manager/src/exploration_fsm.cpp`
- 修改：`src/planner/exploration_manager/src/exploration_manager.cpp`
- 测试：`tests/test_mode_gated_mtsp_policy.py`
- 测试：`tests/test_two_agent_topology.py`

**接口：**

- `bool ExplorationFSM::planAgentsForCycle()`：在两个 Agent 同一轮计划时决定联合或独立路径。

- [ ] **步骤 1：写失败测试**

```python
def test_fsm_uses_one_joint_plan_only_for_matching_modes():
    source = read("src/planner/exploration_manager/src/exploration_fsm.cpp")
    for token in ("planAgentsForCycle", "evaluateNavigationMode", "planJointModeTargets", "mode0 != mode1"):
        assert token in source

def test_normal_planning_does_not_use_frontier_claims():
    manager = read("src/planner/exploration_manager/src/exploration_manager.cpp")
    fsm = read("src/planner/exploration_manager/src/exploration_fsm.cpp")
    assert "claimFrontierByPosition" not in manager
    assert "releaseClaimByAgent" not in fsm
```

- [ ] **步骤 2：验证失败**

运行：`pytest tests/test_mode_gated_mtsp_policy.py -k "one_joint_plan or does_not_use_frontier_claims" -v`

预期：因当前逐 Agent 规划和 Claim 生命周期而失败。

- [ ] **步骤 3：最小实现**

将同一 `PLAN_ACTION` 周期的两台可重规划 Agent 汇集到 `planAgentsForCycle()`。共同模式、候选至少两个且联合求解有效时，同时写入两个首目标；其余情况调用无 Claim 的独立规划。删除候选 Claim 过滤、`claimFrontierByPosition` 和 `releaseClaimByAgent` 的常规调用。

- [ ] **步骤 4：验证通过**

运行：`pytest tests/test_mode_gated_mtsp_policy.py tests/test_two_agent_topology.py -v`

预期：通过。

- [ ] **步骤 5：提交**

运行：`git add src/planner/exploration_manager/include/exploration_manager/exploration_fsm.h src/planner/exploration_manager/src/exploration_fsm.cpp src/planner/exploration_manager/src/exploration_manager.cpp tests/test_mode_gated_mtsp_policy.py tests/test_two_agent_topology.py; git commit -m "feat: dispatch matching modes through joint MTSP"`

### 任务 5：完整验证

**文件：**

- 测试：`tests/test_mode_gated_mtsp_policy.py`
- 测试：`tests/test_two_agent_topology.py`
- 测试：`tests/test_reach_claim_policy.py`
- 测试：`tests/test_stuck_recovery_policy.py`

- [ ] **步骤 1：运行定向测试**

运行：`pytest tests/test_mode_gated_mtsp_policy.py tests/test_two_agent_topology.py tests/test_reach_claim_policy.py tests/test_stuck_recovery_policy.py -v`

预期：通过。

- [ ] **步骤 2：构建 ROS 包**

运行：`catkin_make --pkg exploration_manager lkh_mtsp_solver`

预期：受影响 C++ 包通过编译。

- [ ] **步骤 3：最终审查**

运行：`git diff --check; git status --short`

预期：无空白错误，`tmp/` 保持未跟踪且未被纳入提交。
