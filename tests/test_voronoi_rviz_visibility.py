"""Regression checks for a responsibility overlay visible above /grid_map/free."""

from pathlib import Path
import unittest

import yaml


class VoronoiRvizVisibilityTests(unittest.TestCase):
    def test_voronoi_samples_cover_the_visualization_stride(self):
        """Catches sparse 0.05m squares being hidden by the free-space map."""
        for path in (
            Path("src/planner/exploration_manager/config/ApexNav.rviz"),
            Path("src/planner/exploration_manager/config/ApexNav_Traj.rviz"),
        ):
            displays = yaml.safe_load(path.read_text())["Visualization Manager"]["Displays"]
            display = next(item for item in displays if item.get("Topic") == "/multi_agent/voronoi_regions")
            self.assertEqual(display["Style"], "Flat Squares", path)
            self.assertGreaterEqual(display["Size (m)"], 0.25, path)
            self.assertGreaterEqual(display["Alpha"], 0.85, path)


if __name__ == "__main__":
    unittest.main()
