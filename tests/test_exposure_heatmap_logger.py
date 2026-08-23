import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from basic_utils.exposure_heatmap_logger import ExposureHeatmapJSONLWriter


class ExposureHeatmapJSONLWriterTests(unittest.TestCase):
    def test_appends_a_complete_update_event(self):
        event = SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda: 12.5)),
            event_type="update",
            scene_id="TbHJrupSAjP",
            episode_id="42",
            step_index=7,
            cluster_id=3,
            observed_label=1,
            best_label=2,
            camera_position=SimpleNamespace(x=1.0, y=2.0, z=0.88),
            camera_yaw=0.5,
            grid_addresses=[11, 12],
            exposure_before=[0.0, 0.5],
            exposure_after=[1.0, 1.0],
            total_exposure=2.0,
            mean_exposure=1.0,
            saturation_ratio=0.5,
            contour_cells=4,
            candidate_count=2,
            best_view_x=3.0,
            best_view_y=4.0,
            best_view_yaw=1.2,
            best_view_gain=0.8,
            best_view_path_cost=2.5,
            best_view_score=0.2,
            detail="accepted_detection",
        )

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "exposure_events.jsonl"
            writer = ExposureHeatmapJSONLWriter(output_path)
            writer.write(event)

            records = [json.loads(line) for line in output_path.read_text().splitlines()]

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["event_type"], "update")
        self.assertEqual(record["scene_id"], "TbHJrupSAjP")
        self.assertEqual(record["cluster_id"], 3)
        self.assertEqual(record["observed_label"], 1)
        self.assertEqual(record["camera_pose"], {"x": 1.0, "y": 2.0, "z": 0.88, "yaw": 0.5})
        self.assertEqual(record["grid_updates"][1], {"address": 12, "before": 0.5, "after": 1.0})
        self.assertEqual(record["best_view"]["score"], 0.2)


if __name__ == "__main__":
    unittest.main()
