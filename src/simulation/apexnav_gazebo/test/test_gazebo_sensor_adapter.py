#!/usr/bin/env python3

from collections import deque
import importlib.util
from pathlib import Path
import threading
import unittest
from unittest import mock

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gazebo_sensor_adapter.py"
SPEC = importlib.util.spec_from_file_location("gazebo_sensor_adapter", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
GazeboSensorAdapter = MODULE.GazeboSensorAdapter


class GazeboSensorAdapterTest(unittest.TestCase):
    @staticmethod
    def _time(seconds):
        return rospy.Time.from_sec(seconds)

    def setUp(self):
        self.adapter = GazeboSensorAdapter.__new__(GazeboSensorAdapter)
        self.adapter.world_frame = "map"
        self.adapter.base_frame = "base_link"
        self.adapter.max_future = 0.02
        self.adapter.max_frame_age = 0.50
        self.adapter.history_sec = 5.0
        self.adapter.lock = threading.Lock()
        self.adapter.odom = deque()
        self.adapter.drop_counts = {}
        self.adapter.last_clock = rospy.Time(0)
        self.adapter.last_stamp = rospy.Time(0)
        self.adapter.remap_key = None
        self.adapter.maps = None

    def _odom(self, stamp=10.0):
        msg = Odometry()
        msg.header.stamp = self._time(stamp)
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link"
        msg.pose.pose.orientation.w = 1.0
        return msg

    def test_accepts_finite_current_map_to_base_pose(self):
        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(10.1)):
            self.adapter._odom_cb(self._odom())
        self.assertEqual(len(self.adapter.odom), 1)
        self.assertEqual(self.adapter.drop_counts, {})

    def test_rejects_frame_nonfinite_quaternion_and_source_time_errors(self):
        cases = []
        wrong_frame = self._odom()
        wrong_frame.header.frame_id = "odom"
        cases.append((wrong_frame, "odom_frame_mismatch"))
        nonfinite = self._odom()
        nonfinite.pose.pose.position.z = float("nan")
        cases.append((nonfinite, "nonfinite_odom_pose"))
        invalid_q = self._odom()
        invalid_q.pose.pose.orientation.w = 0.0
        cases.append((invalid_q, "invalid_odom_quaternion"))
        stale = self._odom(9.0)
        cases.append((stale, "odom_source_age"))
        future = self._odom(10.2)
        cases.append((future, "odom_source_age"))

        for msg, reason in cases:
            with self.subTest(reason=reason):
                self.adapter.odom.clear()
                self.adapter.drop_counts.clear()
                with mock.patch.object(
                        MODULE.rospy.Time, "now", return_value=self._time(10.1)):
                    self.adapter._odom_cb(msg)
                self.assertEqual(len(self.adapter.odom), 0)
                self.assertEqual(self.adapter.drop_counts.get(reason), 1)

    def test_clock_rollback_clears_old_history_and_accepts_new_epoch(self):
        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(10.1)):
            self.adapter._odom_cb(self._odom(10.0))
        self.adapter.last_stamp = self._time(10.0)

        with mock.patch.object(MODULE.rospy.Time, "now", return_value=self._time(2.1)):
            self.adapter._odom_cb(self._odom(2.0))

        self.assertEqual(len(self.adapter.odom), 1)
        self.assertEqual(self.adapter.odom[0].header.stamp, self._time(2.0))
        self.assertTrue(self.adapter.last_stamp.is_zero())

    def test_camera_info_requires_finite_nonempty_frame_contract(self):
        info = CameraInfo()
        info.header.frame_id = "camera_optical_frame"
        info.width = 640
        info.height = 480
        info.K = [388.0, 0.0, 320.0, 0.0, 422.0, 240.0, 0.0, 0.0, 1.0]
        info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.P = [388.0, 0.0, 320.0, 0.0, 0.0, 422.0,
                  240.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.assertTrue(self.adapter._camera_info_valid(info))
        info.K[0] = float("nan")
        self.assertFalse(self.adapter._camera_info_valid(info))
        info.K[0] = 388.0
        info.header.frame_id = ""
        self.assertFalse(self.adapter._camera_info_valid(info))


if __name__ == "__main__":
    unittest.main()
