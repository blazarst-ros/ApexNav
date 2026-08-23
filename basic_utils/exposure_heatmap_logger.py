"""Persistent JSONL recording for ObjectMap exposure heatmap events."""

import json
from pathlib import Path


class ExposureHeatmapJSONLWriter:
    """Append ROS exposure events as self-contained JSONL records."""

    def __init__(self, output_path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _stamp_seconds(event):
        stamp = getattr(getattr(event, "header", None), "stamp", None)
        return stamp.to_sec() if stamp is not None else 0.0

    def write(self, event):
        grid_updates = [
            {"address": address, "before": before, "after": after}
            for address, before, after in zip(
                event.grid_addresses, event.exposure_before, event.exposure_after
            )
        ]
        record = {
            "event_type": event.event_type,
            "stamp_sec": self._stamp_seconds(event),
            "scene_id": event.scene_id,
            "episode_id": event.episode_id,
            "step_index": event.step_index,
            "cluster_id": event.cluster_id,
            "observed_label": event.observed_label,
            "best_label": event.best_label,
            "camera_pose": {
                "x": event.camera_position.x,
                "y": event.camera_position.y,
                "z": event.camera_position.z,
                "yaw": event.camera_yaw,
            },
            "grid_updates": grid_updates,
            "total_exposure": event.total_exposure,
            "mean_exposure": event.mean_exposure,
            "saturation_ratio": event.saturation_ratio,
            "contour_cells": event.contour_cells,
            "candidate_count": event.candidate_count,
            "best_view": {
                "x": event.best_view_x,
                "y": event.best_view_y,
                "yaw": event.best_view_yaw,
                "gain": event.best_view_gain,
                "path_cost": event.best_view_path_cost,
                "score": event.best_view_score,
            },
            "detail": event.detail,
        }
        with self.output_path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
