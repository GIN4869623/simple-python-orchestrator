import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from simple_orchestrator.config import load_dotenv


class ConfigTests(unittest.TestCase):
    def test_loads_values_from_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            env_path = Path(temporary_directory) / ".env"
            env_path.write_text(
                "OLLAMA_MODEL=local-model\nOLLAMA_TEMPERATURE=0.4\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                loaded_path = load_dotenv(env_path)

                self.assertEqual(env_path, loaded_path)
                self.assertEqual("local-model", os.environ["OLLAMA_MODEL"])
                self.assertEqual("0.4", os.environ["OLLAMA_TEMPERATURE"])

    def test_real_environment_variable_overrides_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            env_path = Path(temporary_directory) / ".env"
            env_path.write_text("OLLAMA_MODEL=file-model\n", encoding="utf-8")
            with patch.dict(
                os.environ, {"OLLAMA_MODEL": "shell-model"}, clear=True
            ):
                load_dotenv(env_path)

                self.assertEqual("shell-model", os.environ["OLLAMA_MODEL"])


if __name__ == "__main__":
    unittest.main()
