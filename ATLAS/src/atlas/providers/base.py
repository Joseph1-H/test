"""The model-provider interface. Every backend (Ollama, OpenAI-compatible, ...) implements this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]
TokenCallback = Callable[[str], None]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class ChatResult:
    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class ProviderError(RuntimeError):
    """A provider failed in a way the user should hear about (unreachable, bad key, ...)."""


class ModelProvider(ABC):
    #: Short name shown to the user, e.g. "ollama".
    name: str
    #: True if inference happens on this machine (no data leaves it).
    is_local: bool

    def __init__(self, model: str) -> None:
        self.model = model

    @property
    def label(self) -> str:
        where = "local" if self.is_local else "cloud"
        return f"{self.model} ({self.name}, {where})"

    @abstractmethod
    def chat(self, messages: Sequence[Message], on_token: TokenCallback | None = None) -> ChatResult:
        """Send the conversation and return the reply, streaming tokens to `on_token`."""

    @abstractmethod
    def list_models(self) -> list[str]:
        """Model names available from this provider."""

    @abstractmethod
    def health(self) -> str:
        """Return a short status string if reachable; raise ProviderError otherwise."""
