import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from llm.answer_reader.answer_reader import read_answer


CONFIG_PATHS = (
    Path("config/habitat_eval_hm3dv1.yaml"),
    Path("config/habitat_eval_hm3dv2.yaml"),
    Path("config/habitat_eval_mp3d.yaml"),
)


class LiteOfflineMultiAgentContractTests(unittest.TestCase):
    def test_offline_reader_uses_valid_cached_answer_without_calling_client(self):
        """Catches offline mode escaping to an LLM despite a usable local answer."""
        with tempfile.TemporaryDirectory() as temp_dir:
            answer_path = Path(temp_dir) / "answers.txt"
            response_path = Path(temp_dir) / "responses.txt"
            answer_path.write_text(
                "chair: ['stool', 'couch', 0.45, 'living room']\n",
                encoding="utf-8",
            )

            with patch(
                "llm.answer_reader.answer_reader.get_answer",
                side_effect=AssertionError("offline mode must not call an LLM client"),
            ):
                result = read_answer(
                    str(answer_path),
                    str(response_path),
                    "chair",
                    SimpleNamespace(llm_client="offline"),
                )

        self.assertEqual(result, (["stool", "couch"], "living room", 0.45))

    def test_offline_reader_returns_neutral_defaults_for_unknown_label_without_client(self):
        """Catches offline unknown labels creating network requests or invented confusions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            answer_path = Path(temp_dir) / "answers.txt"
            response_path = Path(temp_dir) / "responses.txt"

            with patch(
                "llm.answer_reader.answer_reader.get_answer",
                side_effect=AssertionError("offline mode must not call an LLM client"),
            ):
                result = read_answer(
                    str(answer_path),
                    str(response_path),
                    "unseen target",
                    SimpleNamespace(llm_client="offline"),
                )

            self.assertEqual(result, ([], "unknown", 0.4))
            self.assertFalse(answer_path.exists())
            self.assertFalse(response_path.exists())

    def test_eval_configs_select_lite_services_without_changing_two_agent_contract(self):
        """Catches a Lite config that regresses topology, scheduling, or ObjectNav semantics."""
        for config_path in CONFIG_PATHS:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

            self.assertEqual(config["num_agents"], 2, config_path)
            self.assertEqual(config["habitat"]["simulator"]["type"], "MultiAgentSim-v0", config_path)
            self.assertEqual(
                config["habitat"]["simulator"]["agents_order"], ["agent_0", "agent_1"], config_path
            )
            self.assertEqual(
                [config["habitat"]["simulator"]["agents"][name]["height"] for name in ("agent_0", "agent_1")],
                [0.8, 1.5],
                config_path,
            )
            self.assertEqual(config["multiagent"]["perception_agents_per_step"], 3, config_path)
            self.assertEqual(config["multiagent"]["perception_interval_steps"], 1, config_path)
            self.assertEqual(config["multiagent"]["episode_termination"], "cooperative", config_path)
            self.assertEqual(
                config["habitat"]["task"]["measurements"]["success"]["success_distance"],
                0.35,
                config_path,
            )
            self.assertIn("objectnav", config["habitat"]["dataset"]["data_path"], config_path)

            self.assertEqual(set(config["detector"]), {"yoloe"}, config_path)
            self.assertEqual(config["detector"]["yoloe"], {
                "confidence_threshold": 0.3,
                "iou_threshold": 0.5,
                "agnostic_nms": True,
            }, config_path)
            self.assertEqual(config["llm"]["llm_client"]["llm_client"], "offline", config_path)

    def test_multiagent_entrypoints_keep_per_agent_lite_ros_contracts(self):
        """Catches entrypoints collapsing both agents onto legacy single-agent topic names."""
        for entrypoint in (
            Path("habitat_evaluation.py"),
            Path("habitat_manual_control_multiagent.py"),
        ):
            source = entrypoint.read_text(encoding="utf-8")
            self.assertIn(
                "from vlm.utils.get_object_utils import get_object, get_object_class_names",
                source,
                entrypoint,
            )
            self.assertIn('f"/blip2/{agent_name}/cosine_score"', source, entrypoint)
            self.assertIn('f"/detector/{agent_name}/clouds_with_scores"', source, entrypoint)
            self.assertIn("cld_msg.label_indices = label_list", source, entrypoint)
            self.assertIn("get_object_class_names(label, llm_answer)", source, entrypoint)


if __name__ == "__main__":
    unittest.main()
