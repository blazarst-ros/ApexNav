from pathlib import Path


OBJECT_MAP_CPP = Path("src/planner/plan_env/src/object_map2d.cpp")
OBJECT_MAP_H = Path("src/planner/plan_env/include/plan_env/object_map2d.h")
MAP_ROS_CPP = Path("src/planner/plan_env/src/map_ros.cpp")
MANAGER_CPP = Path("src/planner/exploration_manager/src/exploration_manager.cpp")


def test_nonprotection_branch_removes_negative_evidence_fusion():
    """The comparison branch must only fuse actual positive detections."""
    source = OBJECT_MAP_CPP.read_text(encoding="utf-8")
    header = OBJECT_MAP_H.read_text(encoding="utf-8")

    assert "inputObservationObjectsCloud" not in header
    assert "inputObservationObjectsCloud" not in source
    assert "shouldSuppressTargetNegativeEvidence" not in header
    assert "shouldSuppressTargetNegativeEvidence" not in source
    assert "fusionConfidenceScore" in source


def test_map_ros_does_not_extract_undetected_object_observations():
    header = Path("src/planner/plan_env/include/plan_env/map_ros.h").read_text(encoding="utf-8")
    source = MAP_ROS_CPP.read_text(encoding="utf-8")

    assert "getObservationObjectsCloud" not in header
    assert "getObservationObjectsCloud" not in source


def test_object_astar_uses_four_tenths_of_a_second_per_attempt():
    source = MANAGER_CPP.read_text(encoding="utf-8")

    assert "kObjectAstarMaxSearchTime = 0.4" in source


def test_episode_fusion_threshold_is_fixed_at_four_tenths():
    source = Path("habitat_evaluation.py").read_text(encoding="utf-8")

    assert "fusion_threshold = 0.4" in source
