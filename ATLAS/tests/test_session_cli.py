import io
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from atlas.cli import AtlasCLI, main
from atlas.config import Settings
from atlas.providers import OllamaProvider, OpenAICompatProvider, ProviderError
from atlas.session import CloudNotAllowed, Session
from atlas.ui import Style

from fake_servers import FakeServer, unused_port_url


def make_settings(tmp: Path, ollama_url: str, **kw) -> Settings:
    return Settings(workspace=tmp, data_dir=tmp / "data", ollama_host=ollama_url, **kw)


class SessionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_history_kept_and_sent(self):
        with FakeServer("ollama") as srv:
            s = Session(make_settings(self.path, srv.url))
            s.send("first")
            r = s.send("second")
            self.assertEqual(r.content, "echo: second")
            self.assertEqual([m.role for m in s.history], ["user", "assistant", "user", "assistant"])
            sent = srv.requests[-1]["body"]["messages"]
            self.assertEqual(sent[0]["role"], "system")
            self.assertIn("can NOT read files", sent[0]["content"])
            self.assertEqual([m["content"] for m in sent[1:]], ["first", "echo: first", "second"])

    def test_failed_turn_not_saved(self):
        s = Session(make_settings(self.path, unused_port_url()))
        with self.assertRaises(ProviderError):
            s.send("hi")
        self.assertEqual(s.history, [])

    def test_cloud_blocked_by_config(self):
        with FakeServer("ollama") as srv:
            s = Session(make_settings(self.path, srv.url))
            with self.assertRaises(CloudNotAllowed):
                s.enable_cloud(confirm=lambda _: True)
            self.assertEqual(s.mode, "local")

    def test_cloud_requires_user_consent(self):
        with FakeServer("ollama") as osrv, FakeServer("openai") as csrv:
            st = make_settings(self.path, osrv.url, cloud_allowed=True)
            s = Session(st)
            # A public host (127.0.0.1 counts as local), so wrap the fake server URL as remote:
            remote = OpenAICompatProvider("gpt-test", base_url=csrv.url, api_key="secret")
            remote.is_local = False
            asked: list[str] = []
            with self.assertRaises(CloudNotAllowed):
                s.enable_cloud(confirm=lambda p: asked.append(p) or False, cloud_provider=remote)
            self.assertEqual(s.mode, "local")
            self.assertEqual(len(asked), 1)
            self.assertEqual(len(csrv.requests), 0, "nothing may be sent before consent")

            s.enable_cloud(confirm=lambda p: True, cloud_provider=remote)
            self.assertEqual(s.mode, "cloud")
            self.assertEqual(s.send("hi cloud").content, "echo: hi cloud")
            self.assertEqual(len(csrv.requests), 1)

            s.use_local()
            self.assertEqual(s.send("back home").content, "echo: back home")
            self.assertEqual(len(csrv.requests), 1, "local turns must not reach the cloud")


class CLITest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, lines: list[str], settings: Settings) -> str:
        out = io.StringIO()
        inputs = iter(lines)

        def fake_input(_prompt: str) -> str:
            try:
                return next(inputs)
            except StopIteration:
                raise EOFError from None

        cli = AtlasCLI(Session(settings), input_fn=fake_input, out=out, style=Style(out))
        self.assertEqual(cli.run(), 0)
        return out.getvalue()

    def test_full_conversation_and_commands(self):
        with FakeServer("ollama", models=["qwen2.5-coder:7b", "qwen2.5-coder:1.5b"]) as srv:
            st = make_settings(self.path, srv.url)
            out = self.run_cli(
                ["/help", "write hello", "/history", "/model", "/model qwen2.5-coder:1.5b",
                 "/model not-downloaded", "/status", "/cloud", "/clear", "/history", "/bogus", "/exit"],
                st,
            )
        self.assertIn("ATLAS v0.1.0", out)
        self.assertIn("Model: qwen2.5-coder:7b (ollama, local)", out)
        self.assertIn("/history", out)                       # help text
        self.assertIn("ATLAS > echo: write hello", out)       # streamed reply
        self.assertIn("You: write hello", out)                # history
        self.assertIn("* qwen2.5-coder:7b", out)              # model list marks active
        self.assertIn("Now using qwen2.5-coder:1.5b", out)
        self.assertIn("not-downloaded is not downloaded. Run: ollama pull not-downloaded", out)
        self.assertIn("Cloud:     off (not configured)", out)
        self.assertIn("Error: Cloud is turned off", out)
        self.assertIn("Conversation cleared.", out)
        self.assertIn("No messages yet.", out)
        self.assertIn("Unknown command /bogus", out)
        self.assertTrue(out.rstrip().endswith("Bye."))

    def test_cloud_prompt_declined(self):
        with FakeServer("ollama") as osrv, FakeServer("openai") as csrv:
            st = make_settings(self.path, osrv.url, cloud_allowed=True,
                               cloud_base_url="https://api.example.invalid/v1",
                               cloud_model="gpt-test", cloud_api_key="secret")
            out = self.run_cli(["/cloud", "n", "/status"], st)
        self.assertIn("Messages you send will go to api.example.invalid", out)
        self.assertIn("Error: Cloud not enabled. Staying local.", out)
        self.assertIn("Mode:      local", out)
        self.assertEqual(len(csrv.requests), 0)

    def test_provider_error_is_shown_not_crash(self):
        st = make_settings(self.path, unused_port_url())
        out = self.run_cli(["hello"], st)
        self.assertIn("Error: Ollama: cannot connect", out)


class MainEntryTest(unittest.TestCase):
    def test_ask_and_doctor_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp, FakeServer("ollama") as srv:
            env = Path(tmp) / ".env"
            env.write_text(f"OLLAMA_HOST_URL={srv.url}\nATLAS_DATA_DIR={tmp}/data\n")
            old = dict(os.environ)
            os.environ["ATLAS_ENV_FILE"] = str(env)
            for k in ("ATLAS_LOCAL_MODEL", "OLLAMA_HOST_URL", "ATLAS_DATA_DIR"):
                os.environ.pop(k, None)
            buf = io.StringIO()
            try:
                import contextlib
                with contextlib.redirect_stdout(buf):
                    code_ask = main(["--workspace", tmp, "ask", "ping"])
                    code_doc = main(["--workspace", tmp, "doctor"])
            finally:
                os.environ.clear()
                os.environ.update(old)
            out = buf.getvalue()
            self.assertEqual(code_ask, 0)
            self.assertIn("echo: ping", out)
            self.assertEqual(code_doc, 0, out)
            self.assertIn("Ollama server", out)
            self.assertIn("qwen2.5-coder:7b", out)
            log_text = (Path(tmp) / "data" / "atlas.log").read_text()
            self.assertIn('"msg": "chat complete"', log_text)


if __name__ == "__main__":
    unittest.main()
