"""Real HTTP servers that imitate the Ollama and OpenAI APIs, for testing the provider code."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class FakeServer:
    """Start with `with FakeServer(kind) as srv:`; `srv.url` is the base URL, `srv.requests` the log."""

    def __init__(self, kind: str, api_key: str = "secret", models: list[str] | None = None) -> None:
        self.kind = kind  # "ollama" or "openai"
        self.api_key = api_key
        self.models = models or (["qwen2.5-coder:7b", "llama3.2:latest"] if kind == "ollama" else ["gpt-test"])
        self.requests: list[dict[str, Any]] = []
        self.fail_status: int | None = None  # force an HTTP error on chat
        self.reply: str | None = None  # fixed reply instead of echo
        self.truncate = False  # close the stream before the final chunk
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # keep test output quiet
                pass

            def _json(self, code: int, body: Any) -> None:
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                server.requests.append({"method": "GET", "path": self.path, "headers": dict(self.headers)})
                if server.kind == "ollama":
                    if self.path == "/api/version":
                        return self._json(200, {"version": "0.0.test"})
                    if self.path == "/api/tags":
                        return self._json(200, {"models": [{"name": m} for m in server.models]})
                else:
                    if self.path == "/v1/models":
                        if not server._authorized(self):
                            return self._json(401, {"error": {"message": "bad key"}})
                        return self._json(200, {"data": [{"id": m} for m in server.models]})
                self._json(404, {"error": "not found"})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                server.requests.append({"method": "POST", "path": self.path, "body": body,
                                        "headers": dict(self.headers)})
                if server.fail_status:
                    return self._json(server.fail_status, {"error": "forced failure"})
                if server.kind == "ollama" and self.path == "/api/chat":
                    if body.get("model") not in server.models:
                        return self._json(404, {"error": f"model '{body.get('model')}' not found"})
                    return server._stream_ollama(self, body)
                if server.kind == "openai" and self.path == "/v1/chat/completions":
                    if not server._authorized(self):
                        return self._json(401, {"error": {"message": "Incorrect API key"}})
                    return server._stream_openai(self, body)
                self._json(404, {"error": "not found"})

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    # ---- helpers -----------------------------------------------------------------
    @property
    def url(self) -> str:
        host, port = self._httpd.server_address[:2]
        base = f"http://{host}:{port}"
        return base if self.kind == "ollama" else base + "/v1"

    def _authorized(self, handler: BaseHTTPRequestHandler) -> bool:
        return handler.headers.get("Authorization") == f"Bearer {self.api_key}"

    def _reply_for(self, body: dict) -> str:
        if self.reply is not None:
            return self.reply
        last_user = [m["content"] for m in body["messages"] if m["role"] == "user"][-1]
        return f"echo: {last_user}"

    @staticmethod
    def _pieces(text: str) -> list[str]:
        words = text.split(" ")
        return [w + (" " if i < len(words) - 1 else "") for i, w in enumerate(words)]

    def _stream_ollama(self, h: BaseHTTPRequestHandler, body: dict) -> None:
        h.send_response(200)
        h.send_header("Content-Type", "application/x-ndjson")
        h.end_headers()
        for piece in self._pieces(self._reply_for(body)):
            chunk = {"model": body["model"], "message": {"role": "assistant", "content": piece}, "done": False}
            h.wfile.write((json.dumps(chunk) + "\n").encode())
            h.wfile.flush()
        if not self.truncate:
            final = {"model": body["model"], "message": {"role": "assistant", "content": ""},
                     "done": True, "prompt_eval_count": 12, "eval_count": 7}
            h.wfile.write((json.dumps(final) + "\n").encode())

    def _stream_openai(self, h: BaseHTTPRequestHandler, body: dict) -> None:
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.end_headers()
        h.wfile.write(b": keep-alive\n\n")
        for piece in self._pieces(self._reply_for(body)):
            chunk = {"model": body["model"], "choices": [{"index": 0, "delta": {"content": piece}}]}
            h.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            h.wfile.flush()
        done = {"model": body["model"], "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        h.wfile.write(f"data: {json.dumps(done)}\n\ndata: [DONE]\n\n".encode())

    def __enter__(self) -> "FakeServer":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


def unused_port_url() -> str:
    """A localhost URL where nothing is listening."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}"
