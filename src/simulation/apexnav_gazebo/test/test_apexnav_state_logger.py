#!/usr/bin/env python3

import csv
import importlib.util
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose
from mavros_msgs.msg import State, StatusText
from mavros_msgs.msg import PositionTarget
from nav_msgs.msg import Odometry
from plan_env.msg import MissionState
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Bool, Int32


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "apexnav_state_logger.py"
SPEC = importlib.util.spec_from_file_location("apexnav_state_logger", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ApexNavStateLogger = MODULE.ApexNavStateLogger


class ApexNavStateLoggerTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "state.csv"
        logger = ApexNavStateLogger.__new__(ApexNavStateLogger)
        logger.lock = threading.Lock()
        logger.closed = False
        logger.last_signatures = {}
        logger.last_write_monotonic = {}
        logger.mission_sample_period = 1.0
        logger.state_heartbeat = 5.0
        logger.gazebo_model_name = "iris"
        logger.selected_gazebo_model = ""
        logger.estimator_ground_z = None
        logger.gazebo_ground_z = None
        logger.values = {field: "" for field in logger.FIELDS}
        logger.stream = self.path.open("x", newline="", encoding="utf-8")
        logger.writer = csv.DictWriter(logger.stream, fieldnames=logger.FIELDS)
        logger.writer.writeheader()
        self.logger = logger

    def tearDown(self):
        if not self.logger.closed:
            self.logger.stream.close()
            self.logger.closed = True
        self.temp_dir.cleanup()

    def test_callbacks_write_readable_full_state_snapshots(self):
        now = rospy.Time.from_sec(42.25)
        mission = MissionState(
            state=MissionState.AUTO, target_label="chair", detail="planner active",
            fcu_connected=True, armed=True, px4_mode="OFFBOARD",
            altitude=1.02, planner_active=True)
        mission.header.stamp = rospy.Time.from_sec(42.0)

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=now):
            self.logger._fcu_cb(State(
                connected=True, armed=True, guided=True, mode="OFFBOARD", system_status=4))
            self.logger._mapping_cb(Bool(data=True))
            self.logger._navigation_cb(Bool(data=True))
            self.logger._mission_cb(mission)
            self.logger._planner_cb(Int32(data=3))
            self.logger._result_cb(Int32(data=4))
            status = StatusText(severity=4, text="test warning")
            status.header.stamp = rospy.Time.from_sec(42.1)
            self.logger._status_text_cb(status)
            battery = BatteryState()
            battery.header.stamp = rospy.Time.from_sec(42.2)
            battery.voltage = 15.7
            battery.current = 3.2
            battery.percentage = 0.61
            battery.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
            battery.present = True
            self.logger._battery_cb(battery)

        with self.path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[-1]["event"], "mavros_battery")
        self.assertEqual(rows[-1]["mission_state_name"], "AUTO")
        self.assertEqual(rows[-1]["planner_state_name"], "EXEC_TRAJ")
        self.assertEqual(rows[-1]["exploration_result_name"], "REACH_OBJECT")
        self.assertEqual(rows[-1]["fcu_status_text"], "test warning")
        self.assertEqual(rows[-1]["navigation_enabled"], "True")
        self.assertEqual(rows[-1]["mapping_enabled"], "True")
        self.assertEqual(rows[-1]["battery_voltage"], "15.700000")
        self.assertEqual(rows[-1]["battery_percentage"], "0.610000")
        self.assertEqual(rows[-1]["battery_present"], "True")

    def test_repeated_planner_state_is_deduplicated(self):
        now = rospy.Time.from_sec(5.0)
        with mock.patch.object(MODULE.rospy.Time, "now", return_value=now), \
                mock.patch.object(MODULE.time, "monotonic", return_value=10.0):
            for _ in range(100):
                self.logger._planner_cb(Int32(data=1))

        with self.path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["planner_state_name"], "WAIT_TRIGGER")

    def test_height_sources_and_commanded_z_are_logged_independently(self):
        with mock.patch.object(MODULE.rospy.Time, "now",
                               return_value=rospy.Time.from_sec(20.0)):
            odom = Odometry()
            odom.pose.pose.position.z = -0.48
            self.logger._odom_cb(odom)
            odom.pose.pose.position.z = 0.47
            self.logger._odom_cb(odom)

            models = ModelStates(name=["ground_plane", "iris_0"], pose=[Pose(), Pose()])
            models.pose[1].position.z = 0.13
            self.logger._gazebo_models_cb(models)
            models.pose[1].position.z = 1.10
            self.logger._gazebo_models_cb(models)

            target = PositionTarget()
            target.position.z = 0.52
            target.velocity.z = 0.08
            target.type_mask = 2520
            self.logger._setpoint_cb(target)

        self.assertEqual(self.logger.values["z_feedback_source"],
                         "/mavros/local_position/odom")
        self.assertEqual(self.logger.values["gazebo_model_name"], "iris_0")
        self.assertAlmostEqual(float(self.logger.values["estimator_agl"]), 0.95, places=4)
        self.assertAlmostEqual(float(self.logger.values["gazebo_agl"]), 0.97, places=4)
        self.assertAlmostEqual(float(self.logger.values["gazebo_minus_estimator_agl"]),
                               0.02, places=4)
        self.assertAlmostEqual(float(self.logger.values["setpoint_z"]), 0.52, places=4)
        self.assertAlmostEqual(float(self.logger.values["setpoint_vz"]), 0.08, places=4)
        self.assertEqual(self.logger.values["setpoint_type_mask"], 2520)


if __name__ == "__main__":
    unittest.main()
