from pathlib import Path
import importlib
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import types

import yaml


class TestLLMClientDefaults(unittest.TestCase):
    def test_get_answer_uses_deepseek_without_falling_through(self):
        client = SimpleNamespace(llm_client="deepseek", ollama="unused")

        fake_deepseek_module = types.ModuleType("llm.client.deepseek_answer")
        fake_deepseek_module.deepseek_respond = lambda prompt: "unpatched"
        fake_ollama_module = types.ModuleType("llm.client.ollama_answer")
        fake_ollama_module.ollama_respond = lambda model, prompt: "unpatched"

        with patch.dict(
            sys.modules,
            {
                "llm.client.deepseek_answer": fake_deepseek_module,
                "llm.client.ollama_answer": fake_ollama_module,
            },
        ):
            sys.modules.pop("llm.answer", None)
            llm_answer = importlib.import_module("llm.answer")

        with patch.object(
            llm_answer,
            "deepseek_respond",
            lambda prompt: 'Answer: [chair, couch, 0.30, everywhere]',
        ), patch.object(
            llm_answer,
            "ollama_respond",
            lambda model, prompt: 'Answer: [wrong, branch, 0.10, nowhere]',
        ):
            similar_answer, response = llm_answer.get_answer(client=client, prompt="chair")

        self.assertEqual(response, 'Answer: [chair, couch, 0.30, everywhere]')
        self.assertEqual(similar_answer, ["chair", "couch", 0.30, "everywhere"])

    def test_main_ob_eval_configs_default_to_deepseek(self):
        for config_name in (
            "config/habitat_eval_hm3dv1.yaml",
            "config/habitat_eval_hm3dv2.yaml",
            "config/habitat_eval_mp3d.yaml",
        ):
            config = yaml.safe_load(Path(config_name).read_text())
            self.assertEqual(
                config["llm"]["llm_client"]["llm_client"],
                "deepseek",
                msg=config_name,
            )


if __name__ == "__main__":
    unittest.main()
