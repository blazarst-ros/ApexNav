import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ExposureHeatmapContractTests(unittest.TestCase):
    def test_detection_message_carries_observation_context(self):
        message = (ROOT / "src/planner/plan_env/msg/MultipleMasksWithConfidence.msg").read_text()
        for field in (
            "std_msgs/Header header",
            "string scene_id",
            "string episode_id",
            "uint32 step_index",
            "geometry_msgs/Point camera_position",
            "float64 camera_yaw",
        ):
            self.assertIn(field, message)

    def test_exposure_event_has_increment_and_viewpoint_fields(self):
        message_path = ROOT / "src/planner/plan_env/msg/ExposureHeatmapEvent.msg"
        self.assertTrue(message_path.exists())
        message = message_path.read_text()
        for field in (
            "string event_type",
            "uint32[] grid_addresses",
            "float32[] exposure_before",
            "float32[] exposure_after",
            "float64 best_view_score",
        ):
            self.assertIn(field, message)

    def test_object_map_owns_shared_exposure_and_explicit_logs(self):
        header = (ROOT / "src/planner/plan_env/include/plan_env/object_map2d.h").read_text()
        source = (ROOT / "src/planner/plan_env/src/object_map2d.cpp").read_text()
        planner = (ROOT / "src/planner/exploration_manager/src/exploration_manager.cpp").read_text()
        self.assertIn("exposure_by_grid_", header)
        self.assertIn("updateExposureHeatmap", header)
        self.assertIn("[ExposureHeatmap][UPDATE]", source)
        self.assertIn("[ExposureHeatmap][RESET]", source)
        self.assertIn("consumeExposureViewCandidates", source)
        self.assertIn("[ExposureHeatmap][RANK]", planner)
        self.assertIn("computePathCost", planner)
        self.assertNotIn("if (label == 0) {\n    updateExposureHeatmap", source)

    def test_python_populates_context_and_subscribes_for_jsonl(self):
        source = (ROOT / "habitat_evaluation.py").read_text()
        for statement in (
            "cld_with_score_msg.scene_id = env.current_episode.scene_id",
            "cld_with_score_msg.episode_id = str(env.current_episode.episode_id)",
            "cld_with_score_msg.step_index = count_steps",
            "ExposureHeatmapJSONLWriter",
            'rospy.Subscriber("/object/exposure_events"',
        ):
            self.assertIn(statement, source)

    def test_rviz_has_a_separate_exposure_layer(self):
        for config_name in ("ApexNav.rviz", "ApexNav_Traj.rviz"):
            rviz = (ROOT / "src/planner/exploration_manager/config" / config_name).read_text()
            self.assertIn("/object/exposure_heatmap", rviz, config_name)


if __name__ == "__main__":
    unittest.main()
