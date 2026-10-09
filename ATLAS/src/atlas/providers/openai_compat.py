"""Any OpenAI-compatible /chat/completions endpoint (cloud APIs, vLLM, llama.cpp server, TGI, ...)."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from urllib.parse import urlparse

from .base import ChatResult, Message, ModelProvider, ProviderError, TokenCallback
from .http import get_json, stream_lines

log = logging.getLogger("atlas.providers.openai")

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


class OpenAICompatProvider(ModelProvider):
    name = "openai-compatible"

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "",
        temperature: float = 0.2,
        timeout: float = 300,
    ) -> None:
        super().__init__(model)
        if not base_url.startswith(("http://", "https://")):
            raise ProviderError(f"Invalid base URL {base_url!r}: it must start with http:// or https://")
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        # A server on this machine (e.g. vLLM or llama.cpp serving a Hugging Face model) is local.
        self.is_local = (urlparse(self.base_url).hostname or "") in _LOCAL_HOSTS

    @property
    def host(self) -> str:
        return urlparse(self.base_url).hostname or self.base_url

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def chat(self, messages: Sequence[Message], on_token: TokenCallback | None = None) -> ChatResult:
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
            "temperature": self.temperature,
        }
        parts: list[str] = []
        model_name = self.model
        finished = False
        for line in stream_lines(
            f"{self.base_url}/chat/completions", payload, self.timeout, self.host, self._headers()
        ):
            if not line.startswith("data:"):
                continue  # SSE comments / keep-alives
            data = line[5:].strip()
            if data == "[DONE]":
                finished = True
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError as err:
                raise ProviderError(f"{self.host} sent an unreadable response: {data[:120]!r}") from err
            if "error" in chunk:
                err_obj = chunk["error"]
                msg = err_obj.get("message", err_obj) if isinstance(err_obj, dict) else err_obj
                raise ProviderError(f"{self.host}: {msg}")
            model_name = chunk.get("model", model_name)
            for choice in chunk.get("choices", []):
                piece = (choice.get("delta") or {}).get("content") or ""
                if piece:
                    parts.append(piece)
                    if on_token:
                        on_token(piece)
                if choice.get("finish_reason"):
                    finished = True
        if not finished and not parts:
            raise ProviderError(f"{self.host} closed the connection without a reply.")
        log.info("chat complete", extra={"provider": self.name, "model": model_name, "host": self.host})
        return ChatResult(content="".join(parts), model=model_name)

    def list_models(self) -> list[str]:
        data = get_json(f"{self.base_url}/models", 15, self.host, self._headers())
        return sorted(m["id"] for m in data.get("data", []) if "id" in m)

    def health(self) -> str:
        count = len(self.list_models())
        return f"{self.host} reachable ({count} models)"
