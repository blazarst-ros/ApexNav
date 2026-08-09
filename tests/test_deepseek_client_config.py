import importlib
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


class TestDeepSeekClientConfig(unittest.TestCase):
    def test_get_api_key_reads_environment(self):
        fake_openai = types.ModuleType("openai")

        class DummyOpenAI:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_openai.OpenAI = DummyOpenAI

        old_value = os.environ.get("DEEPSEEK_API_KEY")
        os.environ["DEEPSEEK_API_KEY"] = "test-key"

        try:
            with unittest.mock.patch.dict(sys.modules, {"openai": fake_openai}):
                sys.modules.pop("llm.client.deepseek_answer", None)
                module = importlib.import_module("llm.client.deepseek_answer")
                self.assertEqual(module._get_api_key(), "test-key")
        finally:
            if old_value is None:
                os.environ.pop("DEEPSEEK_API_KEY", None)
            else:
                os.environ["DEEPSEEK_API_KEY"] = old_value

    def test_get_api_key_reads_local_env_file(self):
        fake_openai = types.ModuleType("openai")

        class DummyOpenAI:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_openai.OpenAI = DummyOpenAI

        old_value = os.environ.pop("DEEPSEEK_API_KEY", None)

        try:
            with unittest.mock.patch.dict(sys.modules, {"openai": fake_openai}):
                sys.modules.pop("llm.client.deepseek_answer", None)
                module = importlib.import_module("llm.client.deepseek_answer")
                with tempfile.TemporaryDirectory() as tmpdir:
                    env_file = Path(tmpdir) / ".env.local"
                    env_file.write_text("DEEPSEEK_API_KEY=file-key\n")
                    self.assertEqual(module._get_api_key(env_file=env_file), "file-key")
        finally:
            if old_value is not None:
                os.environ["DEEPSEEK_API_KEY"] = old_value


if __name__ == "__main__":
    unittest.main()
