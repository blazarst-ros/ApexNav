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
        assert "constexpr double REACH_DISTANCE = 0.20;" in text
        assert "constexpr double SOFT_REACH_DISTANCE = 0.20;" in text


def test_eval_configs_match_main_ob_habitat_success_distance():
    for config_path in [
        Path("config/habitat_eval_hm3dv1.yaml"),
        Path("config/habitat_eval_hm3dv2.yaml"),
        Path("config/habitat_eval_mp3d.yaml"),
        Path("config/habitat_vel_control.yaml"),
    ]:
        assert "success_distance: 0.35" in config_path.read_text(encoding="utf-8")
