"""Local inference through Ollama's HTTP API (https://github.com/ollama/ollama/blob/main/docs/api.md)."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence

from .base import ChatResult, Message, ModelProvider, ProviderError, TokenCallback
from .http import get_json, stream_lines

log = logging.getLogger("atlas.providers.ollama")


class OllamaProvider(ModelProvider):
    name = "ollama"
    is_local = True

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        num_ctx: int = 8192,
        temperature: float = 0.2,
        timeout: float = 300,
    ) -> None:
        super().__init__(model)
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout = timeout

    def chat(self, messages: Sequence[Message], on_token: TokenCallback | None = None) -> ChatResult:
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
            "options": {"num_ctx": self.num_ctx, "temperature": self.temperature},
        }
        parts: list[str] = []
        final: dict = {}
        for line in stream_lines(f"{self.host}/api/chat", payload, self.timeout, "Ollama"):
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as err:
                raise ProviderError(f"Ollama sent an unreadable response: {line[:120]!r}") from err
            if "error" in chunk:
                raise ProviderError(f"Ollama: {chunk['error']}")
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                parts.append(piece)
                if on_token:
                    on_token(piece)
            if chunk.get("done"):
                final = chunk
                break
        else:
            if not final:
                raise ProviderError("Ollama closed the connection before finishing the reply.")

        result = ChatResult(
            content="".join(parts),
            model=final.get("model", self.model),
            prompt_tokens=final.get("prompt_eval_count"),
            completion_tokens=final.get("eval_count"),
        )
        log.info(
            "chat complete",
            extra={"provider": self.name, "model": result.model,
                   "prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens},
        )
        return result

    def list_models(self) -> list[str]:
        data = get_json(f"{self.host}/api/tags", 15, "Ollama")
        return sorted(m["name"] for m in data.get("models", []) if "name" in m)

    def health(self) -> str:
        data = get_json(f"{self.host}/api/version", 5, "Ollama")
        return f"Ollama {data.get('version', '?')} at {self.host}"

    def has_model(self, model: str | None = None) -> bool:
        wanted = model or self.model
        names = self.list_models()
        # "qwen2.5-coder" matches "qwen2.5-coder:latest"
        return wanted in names or f"{wanted}:latest" in names
