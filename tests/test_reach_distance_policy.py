from pathlib import Path


DISCRETE_FSM_HEADER = Path(
    "src/planner/exploration_manager/include/exploration_manager/exploration_fsm.h"
)
TRAJ_FSM_HEADER = Path(
    "src/planner/exploration_manager/include/exploration_manager/exploration_fsm_traj.h"
)


def test_cpp_object_reach_thresholds_match_multiagent_success_policy():
    for header_path in [DISCRETE_FSM_HEADER, TRAJ_FSM_HEADER]:
        text = header_path.read_text(encoding="utf-8")
        assert "constexpr double REACH_DISTANCE = 0.50;" in text
        assert "constexpr double SOFT_REACH_DISTANCE = 0.70;" in text
