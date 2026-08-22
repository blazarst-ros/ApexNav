import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from llm.answer_reader.answer_reader import read_answer
from vlm.label_utils import normalize_objectnav_label


CONFIG_PATHS = (
    Path("config/habitat_eval_hm3dv1.yaml"),
    Path("config/habitat_eval_hm3dv2.yaml"),
    Path("config/habitat_eval_mp3d.yaml"),
)


class LiteOfflineMultiAgentContractTests(unittest.TestCase):
    def test_dataset_label_normalization_needs_no_mp3d_metadata_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            answer_path = Path(temp_dir) / "answers.txt"
            response_path = Path(temp_dir) / "responses.txt"
            answer_path.write_text(
                "couch: ['bench', 'chair', 0.55, 'living room']\n",
                encoding="utf-8",
            )

            with patch("gzip.open", side_effect=FileNotFoundError("MP3D metadata absent")):
                hm3d_label = normalize_objectnav_label(
                    "sofa",
                    "data/datasets/objectnav/hm3d/v2/{split}/{split}.json.gz",
                )
                result = read_answer(
                    str(answer_path),
                    str(response_path),
                    hm3d_label,
                    SimpleNamespace(llm_client="offline"),
                )

        self.assertEqual(hm3d_label, "couch")
        self.assertEqual(result, (["bench", "chair"], "living room", 0.55))
        self.assertEqual(
            normalize_objectnav_label(
                "table",
                "data/datasets/objectnav/mp3d/v1/{split}/{split}.json.gz",
            ),
            "table | dining table | coffee table | desk",
        )

    def test_entrypoints_normalize_labels_from_the_configured_dataset(self):
        for entrypoint in (
            Path("habitat_evaluation.py"),
            Path("habitat_manual_control_multiagent.py"),
        ):
            source = entrypoint.read_text(encoding="utf-8")
            self.assertIn(
                "from vlm.label_utils import normalize_objectnav_label",
                source,
                entrypoint,
            )
            self.assertIn(
                "normalize_objectnav_label(label, cfg.habitat.dataset.data_path)",
                source,
                entrypoint,
            )
            self.assertNotIn(
                "data/datasets/objectnav/mp3d/v1/val/val.json.gz",
                source,
                entrypoint,
            )

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

    def test_offline_reader_rejects_poisoned_cache_without_executing_it(self):
        """Catches cache parsing that executes an expression before offline fallback."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            answer_path = temp_path / "answers.txt"
            response_path = temp_path / "responses.txt"
            marker_path = temp_path / "cache-payload-ran"
            answer_path.write_text(
                "unseen target: __import__('pathlib').Path(%r).write_text('ran')\n" % str(marker_path),
                encoding="utf-8",
            )

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
            self.assertFalse(marker_path.exists())
            self.assertFalse(response_path.exists())

    def test_shipped_legacy_cache_values_remain_literal_evaluable(self):
        """Catches a safe cache parser that cannot read the shipped answer literals."""
        legacy_count = 0
        for answer_path in (
            Path("llm/answers/llm_answer_hm3d.txt"),
            Path("llm/answers/llm_answer_mp3d.txt"),
        ):
            for line in answer_path.read_text(encoding="utf-8").splitlines():
                if not line or line.startswith("#"):
                    continue
                _, literal = line.split(": ", 1)
                self.assertIsInstance(ast.literal_eval(literal), list, line)
                legacy_count += 1
        self.assertEqual(legacy_count, 42)

    def test_eval_configs_select_lite_services_without_changing_two_agent_contract(self):
        """Catches a Lite config that regresses topology, scheduling, or ObjectNav semantics."""
        expected_dataset_paths = {
            Path("config/habitat_eval_hm3dv1.yaml"): "data/datasets/objectnav/hm3d/v1/{split}/{split}.json.gz",
            Path("config/habitat_eval_hm3dv2.yaml"): "data/datasets/objectnav/hm3d/v2/{split}/{split}.json.gz",
            Path("config/habitat_eval_mp3d.yaml"): "data/datasets/objectnav/mp3d/v1/{split}/{split}.json.gz",
        }
        expected_agents = {
            "agent_0": {"height": 0.8, "position": [0, 0.8, 0]},
            "agent_1": {"height": 1.5, "position": [0, 1.5, 0]},
        }

        for config_path in CONFIG_PATHS:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            simulator = config["habitat"]["simulator"]

            self.assertEqual(config["num_agents"], 2, config_path)
            self.assertEqual(simulator["type"], "MultiAgentSim-v0", config_path)
            self.assertEqual(simulator["agents_order"], ["agent_0", "agent_1"], config_path)
            for agent_name, expected_agent in expected_agents.items():
                agent = simulator["agents"][agent_name]
                self.assertEqual(agent["height"], expected_agent["height"], config_path)
                self.assertEqual(agent["radius"], 0.18, config_path)
                for sensor_name in ("rgb_sensor", "depth_sensor"):
                    sensor = agent["sim_sensors"][sensor_name]
                    self.assertEqual(sensor["uuid"], f"{agent_name}_{sensor_name[:-7]}", config_path)
                    self.assertEqual(sensor["position"], expected_agent["position"], config_path)

            self.assertEqual(config["habitat"]["environment"]["max_episode_steps"], 250, config_path)
            self.assertEqual(config["multiagent"]["perception_agents_per_step"], 3, config_path)
            self.assertEqual(config["multiagent"]["perception_interval_steps"], 1, config_path)
            self.assertEqual(config["multiagent"]["episode_termination"], "cooperative", config_path)
            self.assertFalse(config["multiagent"]["inter_agent_avoidance"], config_path)
            self.assertEqual(config["multiagent"]["agent_spawn_offset"], 1.0, config_path)
            self.assertEqual(
                config["habitat"]["task"]["measurements"]["success"]["success_distance"],
                0.35,
                config_path,
            )
            self.assertEqual(config["habitat"]["dataset"]["data_path"], expected_dataset_paths[config_path], config_path)

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
