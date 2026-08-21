#!/usr/bin/env python3
"""Regression tests for the PX4 Iris 2-D obstacle height slice."""

from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import yaml


PACKAGE = Path(__file__).resolve().parents[1]
WORKSPACE_SRC = PACKAGE.parents[1]


def argument_values(path):
    root = ET.parse(str(path)).getroot()
    return {
        element.attrib["name"]: element.attrib.get("value", element.attrib.get("default"))
        for element in root.iter("arg")
        if "name" in element.attrib
    }


class GazeboHeightFilterLaunchTest(unittest.TestCase):
    def test_gazebo_launches_use_aircraft_relative_height_slice(self):
        for name in ("gazebo_planner.launch", "gazebo_full_stack.launch"):
            values = argument_values(PACKAGE / "launch" / name)
            self.assertEqual(values["height_filter_reference"], "sensor")
            self.assertAlmostEqual(float(values["filter_min_height"]), -0.18)
            self.assertAlmostEqual(float(values["filter_max_height"]), 0.25)
            self.assertLess(float(values["filter_min_height"]), 0.0)
            self.assertGreater(float(values["filter_max_height"]), 0.0)

    def test_height_reference_reaches_map_ros_parameter(self):
        exploration_launch = (
            WORKSPACE_SRC / "planner" / "exploration_manager" /
            "launch" / "exploration_traj.launch")
        algorithm_launch = exploration_launch.with_name("algorithm_traj.xml")
        exploration_text = exploration_launch.read_text(encoding="utf-8")
        algorithm_text = algorithm_launch.read_text(encoding="utf-8")
        self.assertIn('name="height_filter_reference" default="world"', exploration_text)
        self.assertIn(
            'name="height_filter_reference_" value="$(arg height_filter_reference)"',
            exploration_text)
        self.assertIn(
            'name="map_ros/height_filter_reference" '
            'value="$(arg height_filter_reference_)"', algorithm_text)

    def test_global_clearance_is_backed_by_runtime_oriented_footprint_check(self):
        """A modest global halo is safe only with the exact runtime footprint guard."""
        planning_values = {}
        for line in (PACKAGE / "config" / "iris_planning_param.yaml").read_text(
                encoding="utf-8").splitlines():
            if ":" in line and not line.lstrip().startswith("#"):
                key, value = line.split(":", 1)
                planning_values[key.strip()] = value.strip()
        half_length = float(planning_values["length"]) / 2.0
        half_width = float(planning_values["width"]) / 2.0
        circumscribed_radius = (half_length ** 2 + half_width ** 2) ** 0.5

        for name in ("gazebo_planner.launch", "gazebo_full_stack.launch"):
            values = argument_values(PACKAGE / "launch" / name)
            clearance = float(values["obstacles_inflation"])
            self.assertGreaterEqual(clearance, max(half_length, half_width))
            self.assertLessEqual(clearance, circumscribed_radius)

        fsm_source = (WORKSPACE_SRC / "planner" / "exploration_manager" /
                      "src" / "exploration_fsm_traj.cpp").read_text(encoding="utf-8")
        kino_source = (WORKSPACE_SRC / "planner" / "path_searching" /
                       "src" / "kino_astar.cpp").read_text(encoding="utf-8")
        self.assertIn("isCollisionPosYaw(check_pos_2d, check_yaw)", fsm_source)
        self.assertIn("complete oriented footprint", kino_source)

    def test_kinoastar_yaw_resolution_is_explicit(self):
        """Catches the historical uninitialized yaw discretization member."""
        config = (PACKAGE / "config" / "iris_planning_param.yaml").read_text(
            encoding="utf-8")
        values = {}
        for line in config.splitlines():
            if ":" in line and not line.lstrip().startswith("#"):
                key, value = line.split(":", 1)
                values[key.strip()] = value.strip()

        self.assertIn("yaw_resolution", values)
        self.assertGreater(float(values["yaw_resolution"]), 0.0)

    def test_supervisor_watchdogs_and_px4_loss_action_are_explicit(self):
        config = yaml.safe_load((PACKAGE / "config" / "px4_supervisor.yaml").read_text(
            encoding="utf-8"))
        self.assertGreater(config["first_reference_timeout_sec"],
                           config["command_timeout_sec"])
        self.assertEqual(config["px4_expected_com_obl_rc_act"], 4)
        self.assertEqual(config["px4_expected_com_rc_in_mode"], 4)
        self.assertAlmostEqual(config["px4_expected_com_of_loss_t"], 1.0)
        self.assertEqual(config["world_frame"], "map")
        self.assertEqual(config["base_frame"], "base_link")
        self.assertEqual(config["camera_frame"], "iris_camera_optical_frame")

    def test_perception_enforces_the_same_frame_chain_as_mapping(self):
        source = (PACKAGE.parents[2] / "real_world_test_example" /
                  "real_world_perception.py").read_text(encoding="utf-8")
        self.assertIn("rgb_msg.header.frame_id != self.camera_frame", source)
        self.assertIn("depth_msg.header.frame_id != self.camera_frame", source)
        self.assertIn("pose_msg.header.frame_id != self.world_frame", source)
        self.assertIn("pose_msg.child_frame_id != self.camera_frame", source)
        self.assertIn("quaternion_norm < 0.95", source)

    def test_capture_profile_includes_independent_px4_battery_evidence(self):
        capture = (PACKAGE.parents[2] / "RuntimeData" /
                   "capture_ros_data.sh").read_text(encoding="utf-8")
        self.assertIn("/mavros/battery", capture)

    def test_supervisor_health_uses_only_committed_map_heartbeat(self):
        source = (PACKAGE / "scripts" / "px4_mission_supervisor.py").read_text(
            encoding="utf-8")
        self.assertIn('Subscriber("/grid_map/commit", Header', source)
        self.assertNotIn(
            'Subscriber("/grid_map/filtered_depth_cloud", PointCloud2', source)
        algorithm = (WORKSPACE_SRC / "planner" / "exploration_manager" /
                     "launch" / "algorithm_traj.xml").read_text(encoding="utf-8")
        self.assertIn(
            'name="map_commit_topic_" default="/grid_map/commit"', algorithm)
        self.assertIn(
            'from ="/grid_map/commit" to = "$(arg map_commit_topic_)"', algorithm)

    def test_supervisor_is_the_only_fcu_setpoint_publisher(self):
        publishers = []
        for path in WORKSPACE_SRC.rglob("*"):
            if path.suffix not in (".py", ".cpp", ".h") or "test" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if ('Publisher(\n            "/mavros/setpoint_raw/local"' in text or
                    'advertise<mavros_msgs::PositionTarget>(\n'
                    '        "/mavros/setpoint_raw/local"' in text):
                publishers.append(path)
        self.assertEqual(
            publishers,
            [PACKAGE / "scripts" / "px4_mission_supervisor.py"])


if __name__ == "__main__":
    unittest.main()
