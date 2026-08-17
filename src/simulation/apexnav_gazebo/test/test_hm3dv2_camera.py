#!/usr/bin/env python3

import math
from pathlib import Path
import unittest

import cv2
import numpy as np
import yaml


class HM3DV2CameraTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "config" / "hm3dv2_camera.yaml"
        cls.camera = yaml.safe_load(path.read_text())["camera"]

    def test_intrinsics_match_repository_projection(self):
        camera = self.camera
        fx = camera["width"] / (2.0 * math.tan(math.radians(camera["hfov_deg"] / 2.0)))
        vfov = camera["hfov_deg"] / camera["width"] * camera["height"]
        fy = camera["height"] / (2.0 * math.tan(math.radians(vfov / 2.0)))
        self.assertAlmostEqual(camera["fx"], fx, places=9)
        self.assertAlmostEqual(camera["fy"], fy, places=9)
        self.assertEqual((camera["cx"], camera["cy"]), (320.0, 240.0))

    def test_depth_remap_uses_nearest_and_target_shape(self):
        camera = self.camera
        source_width, source_height = 848, 480
        source_hfov = 1.500983
        source_fx = source_width / (2.0 * math.tan(source_hfov / 2.0))
        source_k = np.array([[source_fx, 0.0, source_width / 2.0],
                             [0.0, source_fx, source_height / 2.0],
                             [0.0, 0.0, 1.0]], dtype=np.float64)
        target_k = np.array([[camera["fx"], 0.0, camera["cx"]],
                             [0.0, camera["fy"], camera["cy"]],
                             [0.0, 0.0, 1.0]], dtype=np.float64)
        maps = cv2.initUndistortRectifyMap(
            source_k, np.zeros(5), np.eye(3), target_k,
            (camera["width"], camera["height"]), cv2.CV_32FC1)
        depth = np.zeros((source_height, source_width), dtype=np.float32)
        depth[:, source_width // 2 :] = 2.0
        remapped = cv2.remap(depth, maps[0], maps[1], cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        self.assertEqual(remapped.shape, (480, 640))
        self.assertTrue(np.all(np.isin(remapped, [0.0, 2.0])))


if __name__ == "__main__":
    unittest.main()
