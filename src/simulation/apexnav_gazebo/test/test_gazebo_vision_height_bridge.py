#!/usr/bin/env python3

import importlib.util
import math
import os
import unittest

from geometry_msgs.msg import Pose


SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts",
                      "gazebo_vision_height_bridge.py")
SPEC = importlib.util.spec_from_file_location("gazebo_vision_height_bridge", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GazeboVisionHeightBridgeTest(unittest.TestCase):
    def test_relative_position_uses_launch_origin(self):
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = 3.0, -2.0, 1.25
        self.assertEqual(MODULE.GazeboVisionHeightBridge.relative_position(
            pose, (1.0, -1.0, 0.25)), (2.0, -1.0, 1.0))

    def test_pose_contract_rejects_nonfinite_and_bad_quaternion(self):
        pose = Pose()
        pose.orientation.w = 1.0
        self.assertTrue(MODULE.GazeboVisionHeightBridge._finite_pose(pose))
        pose.position.z = float("nan")
        self.assertFalse(MODULE.GazeboVisionHeightBridge._finite_pose(pose))
        pose.position.z = 0.0
        pose.orientation.w = 0.2
        self.assertFalse(MODULE.GazeboVisionHeightBridge._finite_pose(pose))


if __name__ == "__main__":
    unittest.main()
