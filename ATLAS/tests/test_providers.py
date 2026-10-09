import unittest

from atlas.providers import Message, OllamaProvider, OpenAICompatProvider, ProviderError

from fake_servers import FakeServer, unused_port_url

CONVO = [Message("system", "sys"), Message("user", "hello world")]


class OllamaProviderTest(unittest.TestCase):
    def test_streams_reply_and_reports_tokens(self):
        with FakeServer("ollama") as srv:
            p = OllamaProvider("qwen2.5-coder:7b", host=srv.url, num_ctx=4096, temperature=0.1)
            tokens: list[str] = []
            result = p.chat(CONVO, on_token=tokens.append)
        self.assertEqual(result.content, "echo: hello world")
        self.assertEqual("".join(tokens), "echo: hello world")
        self.assertGreater(len(tokens), 1, "reply should arrive in several streamed pieces")
        self.assertEqual((result.prompt_tokens, result.completion_tokens), (12, 7))
        sent = srv.requests[-1]["body"]
        self.assertTrue(sent["stream"])
        self.assertEqual(sent["options"], {"num_ctx": 4096, "temperature": 0.1})
        self.assertEqual(sent["messages"][0], {"role": "system", "content": "sys"})

    def test_list_models_health_and_has_model(self):
        with FakeServer("ollama", models=["qwen2.5-coder:7b", "mistral:latest"]) as srv:
            p = OllamaProvider("qwen2.5-coder:7b", host=srv.url)
            self.assertEqual(p.list_models(), ["mistral:latest", "qwen2.5-coder:7b"])
            self.assertIn("0.0.test", p.health())
            self.assertTrue(p.has_model())
            self.assertTrue(p.has_model("mistral"))  # matches mistral:latest
            self.assertFalse(p.has_model("nope:1b"))

    def test_missing_model_is_a_clear_error(self):
        with FakeServer("ollama") as srv:
            p = OllamaProvider("ghost:1b", host=srv.url)
            with self.assertRaises(ProviderError) as ctx:
                p.chat(CONVO)
        self.assertIn("ghost:1b", str(ctx.exception))

    def test_unreachable_server(self):
        p = OllamaProvider("x", host=unused_port_url())
        with self.assertRaises(ProviderError) as ctx:
            p.health()
        self.assertIn("cannot connect", str(ctx.exception))

    def test_server_error(self):
        with FakeServer("ollama") as srv:
            srv.fail_status = 500
            with self.assertRaises(ProviderError) as ctx:
                OllamaProvider("qwen2.5-coder:7b", host=srv.url).chat(CONVO)
        self.assertIn("500", str(ctx.exception))

    def test_stream_cut_off_is_an_error(self):
        with FakeServer("ollama") as srv:
            srv.truncate = True
            with self.assertRaises(ProviderError):
                OllamaProvider("qwen2.5-coder:7b", host=srv.url).chat(CONVO)


class OpenAICompatProviderTest(unittest.TestCase):
    def test_streams_reply_with_auth(self):
        with FakeServer("openai") as srv:
            p = OpenAICompatProvider("gpt-test", base_url=srv.url, api_key="secret")
            tokens: list[str] = []
            result = p.chat(CONVO, on_token=tokens.append)
        self.assertEqual(result.content, "echo: hello world")
        self.assertGreater(len(tokens), 1)
        self.assertEqual(srv.requests[-1]["headers"]["Authorization"], "Bearer secret")

    def test_bad_key(self):
        with FakeServer("openai") as srv:
            p = OpenAICompatProvider("gpt-test", base_url=srv.url, api_key="wrong")
            with self.assertRaises(ProviderError) as ctx:
                p.chat(CONVO)
        self.assertIn("authentication failed", str(ctx.exception))
        self.assertNotIn("wrong", str(ctx.exception), "the key must never appear in errors")

    def test_list_models(self):
        with FakeServer("openai", models=["b-model", "a-model"]) as srv:
            p = OpenAICompatProvider("a-model", base_url=srv.url, api_key="secret")
            self.assertEqual(p.list_models(), ["a-model", "b-model"])

    def test_local_vs_remote_detection(self):
        self.assertTrue(OpenAICompatProvider("m", "http://localhost:8000/v1").is_local)
        self.assertFalse(OpenAICompatProvider("m", "https://api.example.com/v1").is_local)

    def test_rejects_invalid_url(self):
        with self.assertRaises(ProviderError):
            OpenAICompatProvider("m", "api.example.com")


if __name__ == "__main__":
    unittest.main()
