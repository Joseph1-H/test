import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from atlas.config import ConfigError, env_file_is_private, load_settings

CLEAN = {k: v for k, v in os.environ.items() if not k.startswith(("ATLAS_", "OLLAMA_HOST_URL"))}


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.env = self.dir / ".env"

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, text: str = "", **extra_env):
        self.env.write_text(text)
        with mock.patch.dict(os.environ, {**CLEAN, **extra_env}, clear=True):
            return load_settings(workspace=self.dir, env_file=self.env)

    def test_defaults(self):
        s = self.load()
        self.assertEqual(s.local_model, "qwen2.5-coder:7b")
        self.assertEqual(s.ollama_host, "http://localhost:11434")
        self.assertFalse(s.cloud_allowed)
        self.assertFalse(s.cloud_configured)
        self.assertEqual(s.workspace, self.dir.resolve())

    def test_reads_env_file(self):
        s = self.load(
            "ATLAS_LOCAL_MODEL=qwen2.5-coder:1.5b\n"
            "ATLAS_CLOUD_BASE_URL=https://api.example.com/v1/\n"
            "ATLAS_CLOUD_MODEL=big-model\n"
            "ATLAS_CLOUD_API_KEY=sk-test\n"
            "ATLAS_CLOUD_ALLOWED=true\n"
            "ATLAS_NUM_CTX=4096\n"
        )
        self.assertEqual(s.local_model, "qwen2.5-coder:1.5b")
        self.assertEqual(s.cloud_base_url, "https://api.example.com/v1")
        self.assertTrue(s.cloud_configured and s.cloud_allowed)
        self.assertEqual(s.num_ctx, 4096)

    def test_real_environment_wins_over_file(self):
        s = self.load("ATLAS_LOCAL_MODEL=from-file\n", ATLAS_LOCAL_MODEL="from-env")
        self.assertEqual(s.local_model, "from-env")

    def test_api_key_hidden_from_repr(self):
        s = self.load("ATLAS_CLOUD_API_KEY=sk-very-secret\n")
        self.assertNotIn("sk-very-secret", repr(s))

    def test_invalid_values(self):
        for text in ("ATLAS_NUM_CTX=abc\n", "ATLAS_NUM_CTX=10\n", "ATLAS_CLOUD_ALLOWED=maybe\n",
                     "ATLAS_TEMPERATURE=5\n", "ATLAS_LOG_LEVEL=LOUD\n"):
            with self.subTest(text=text), self.assertRaises(ConfigError):
                self.load(text)

    def test_project_env_file_is_ignored(self):
        from atlas import config

        self.env.write_text("ATLAS_LOCAL_MODEL=from-project\n")
        old_cwd = os.getcwd()
        os.chdir(self.dir)
        try:
            with mock.patch.dict(os.environ, CLEAN, clear=True), \
                 mock.patch.object(config, "INSTALL_DIR", self.dir / "nowhere"), \
                 mock.patch.object(config.Path, "home", return_value=self.dir / "nohome"):
                self.assertIsNone(config.find_env_file())
        finally:
            os.chdir(old_cwd)

    def test_env_file_privacy(self):
        self.env.write_text("x=1\n")
        self.env.chmod(0o600)
        self.assertTrue(env_file_is_private(self.env))
        self.env.chmod(0o644)
        self.assertFalse(env_file_is_private(self.env))


if __name__ == "__main__":
    unittest.main()
