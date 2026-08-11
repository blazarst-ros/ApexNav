import subprocess
import tempfile
from pathlib import Path


FSM_SOURCE = Path("src/planner/exploration_manager/src/exploration_fsm.cpp")
POLICY_INCLUDE = Path(
    "src/planner/exploration_manager/include/exploration_manager/fsm_policy.h"
)


def test_motion_outcome_policy_only_flags_failed_forward_actions():
    program = r'''
#include <cassert>
#include <exploration_manager/fsm_policy.h>

int main() {
  using apexnav_planner::isFailedForwardAction;
  assert(isFailedForwardAction(1, 1, 0.01, 0.05));
  assert(!isFailedForwardAction(2, 1, 0.00, 0.05));
  assert(!isFailedForwardAction(4, 1, 0.00, 0.05));
  assert(!isFailedForwardAction(1, 1, 0.06, 0.05));
}
'''
    with tempfile.TemporaryDirectory() as tmpdir:
        source = Path(tmpdir) / "policy_test.cpp"
        binary = Path(tmpdir) / "policy_test"
        source.write_text(program, encoding="utf-8")
        subprocess.run(
            [
                "g++", "-std=c++17",
                "-Isrc/planner/exploration_manager/include",
                str(source), "-o", str(binary),
            ],
            check=True,
        )
        subprocess.run([str(binary)], check=True)


def test_escape_exhaustion_marks_forward_cells_and_replans_without_stuck_terminal():
    source = FSM_SOURCE.read_text(encoding="utf-8")

    assert "setForceOccGrid(current_pos)" not in source
    assert "markForwardCollision(" in source
    assert "ad.replan_flag_ = true;" in source
    assert "Stuck for too long, stopping episode" not in source
    assert "MAX_STUCKING_COUNT" not in source


def test_stationary_rotations_do_not_increment_failed_forward_count():
    source = FSM_SOURCE.read_text(encoding="utf-8")

    assert "isFailedForwardAction(" in source
    assert "if (failed_forward)" in source
    assert "if ((current_pos - last_pos).norm() < stucking_distance)" not in source
