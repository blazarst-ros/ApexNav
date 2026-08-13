from pathlib import Path


MANAGER = Path("src/planner/exploration_manager/src/exploration_manager.cpp")
MANAGER_HEADER = Path(
    "src/planner/exploration_manager/include/exploration_manager/exploration_manager.h"
)
FSM = Path("src/planner/exploration_manager/src/exploration_fsm.cpp")
FSM_HEADER = Path("src/planner/exploration_manager/include/exploration_manager/exploration_fsm.h")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_joint_mtsp_contract_exposes_modes_and_two_agent_assignment():
    header = read(MANAGER_HEADER)

    for token in (
        "NavigationMode",
        "SEARCH_BEST_OBJECT",
        "SEARCH_OVER_DEPTH_OBJECT",
        "SEMANTIC_FRONTIER",
        "GEOMETRIC_FRONTIER",
        "JointAssignment",
        "evaluateNavigationMode",
        "planJointModeTargets",
        "std::array",
    ):
        assert token in header


def test_two_start_minmax_solver_uses_dynamic_programming_not_single_depot_lkh():
    source = read(MANAGER)

    for token in (
        "solveTwoStartMinmax",
        "held_karp",
        "agent_positions[0]",
        "agent_positions[1]",
        "MINMAX",
    ):
        assert token in source

    joint_solver = source.split("solveTwoStartMinmax", 1)[1]
    assert "SALESMEN = 2" not in joint_solver


def test_joint_candidate_pool_is_mode_specific_and_uses_astar_costs():
    source = read(MANAGER)

    for token in (
        "collectJointCandidates",
        "case NavigationMode::SEARCH_BEST_OBJECT",
        "case NavigationMode::SEARCH_OVER_DEPTH_OBJECT",
        "case NavigationMode::SEMANTIC_FRONTIER",
        "case NavigationMode::GEOMETRIC_FRONTIER",
        "computeJointMtspCostMatrix",
        "computePathCost",
        "cost < 10000.0",
        "reachable_by_any",
        "std::isfinite(candidates[j].initial_costs[agent])",
    ):
        assert token in source


def test_readme_describes_the_runtime_joint_assignment_not_a_future_plan():
    readme = read(Path("README.md"))

    assert "当前实现采用候选上限 `K=10`" in readme
    assert "模式一致时，会进入双起点 `MINMAX` 联合分配" in readme


def test_fsm_routes_matching_modes_to_one_joint_plan():
    source = read(FSM)
    header = read(FSM_HEADER)

    assert "planAgentsForCycle" in header
    for token in (
        "planAgentsForCycle",
        "evaluateNavigationMode",
        "planJointModeTargets",
        "mode0 != mode1",
    ):
        assert token in source


def test_normal_planning_does_not_use_frontier_claims():
    manager = read(MANAGER)
    fsm = read(FSM)

    assert "claimFrontierByPosition" not in manager
    assert "releaseClaimByAgent" not in fsm
