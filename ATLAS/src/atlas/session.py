"""A chat session: the active model, temporary conversation history, and cloud authorization."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Literal

from .config import Settings
from .providers import (
    ChatResult,
    Message,
    ModelProvider,
    ProviderError,
    make_cloud_provider,
    make_local_provider,
)
from .providers.base import TokenCallback

log = logging.getLogger("atlas.session")

Mode = Literal["local", "cloud"]
ConfirmFn = Callable[[str], bool]

MAX_HISTORY_MESSAGES = 40

SYSTEM_PROMPT = """You are ATLAS (Automated Tasks, Logic and Software), a local-first AI coding assistant.
The user's workspace is: {workspace}

Current capabilities (version 0.1, Phase 1): conversation only. You can NOT read files, write files,
run commands or see the workspace yet. Never claim you created, edited, ran or tested anything.
When code is needed, show it in fenced code blocks with the intended file path, and say what
the user should run to verify it.

Be precise and concise. Prefer correct, simple, working code. If you are unsure, say so."""


class CloudNotAllowed(RuntimeError):
    """Cloud use was refused by configuration or by the user."""


class Session:
    def __init__(self, settings: Settings, local_provider: ModelProvider | None = None) -> None:
        self.settings = settings
        self.local: ModelProvider = local_provider or make_local_provider(settings)
        self.cloud: ModelProvider | None = None
        self.mode: Mode = "local"
        self.cloud_authorized = False  # per session; never persisted
        self.history: list[Message] = []

    # ---- model selection -------------------------------------------------------
    @property
    def active(self) -> ModelProvider:
        if self.mode == "cloud" and self.cloud is not None:
            return self.cloud
        return self.local

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(workspace=self.settings.workspace)

    def use_local(self) -> None:
        self.mode = "local"
        log.info("switched to local", extra={"model": self.local.model})

    def enable_cloud(self, confirm: ConfirmFn, cloud_provider: ModelProvider | None = None) -> ModelProvider:
        """Switch to the cloud model after explicit user authorization.

        Raises CloudNotAllowed if cloud is disabled in config or the user declines.
        """
        if not self.settings.cloud_allowed:
            raise CloudNotAllowed(
                "Cloud is turned off. Set ATLAS_CLOUD_ALLOWED=true in your .env to allow it."
            )
        provider = cloud_provider or self.cloud or make_cloud_provider(self.settings)
        if provider.is_local:
            # e.g. a vLLM server on localhost: nothing leaves the machine, no consent needed.
            self.cloud, self.mode = provider, "cloud"
            return provider

        host = getattr(provider, "host", "the cloud provider")
        prompt = (
            f"Messages you send will go to {host} ({provider.model}).\n"
            f"This includes the current conversation ({len(self.history)} messages).\n"
            "ATLAS never sends project files to the cloud on its own. Allow cloud for this session?"
        )
        if not self.cloud_authorized and not confirm(prompt):
            raise CloudNotAllowed("Cloud not enabled. Staying local.")
        self.cloud_authorized = True
        self.cloud, self.mode = provider, "cloud"
        log.info("switched to cloud", extra={"model": provider.model, "host": host})
        return provider

    def set_model(self, model: str) -> None:
        """Change the model of the active provider."""
        self.active.model = model
        log.info("model changed", extra={"mode": self.mode, "model": model})

    # ---- conversation ----------------------------------------------------------
    def clear(self) -> None:
        self.history.clear()

    def send(self, text: str, on_token: TokenCallback | None = None) -> ChatResult:
        text = text.strip()
        if not text:
            raise ValueError("Message is empty.")
        if self.mode == "cloud" and not self.active.is_local and not self.cloud_authorized:
            raise CloudNotAllowed("Cloud use has not been authorized in this session.")

        user = Message("user", text)
        messages = [Message("system", self.system_prompt), *self.history[-MAX_HISTORY_MESSAGES:], user]
        try:
            result = self.active.chat(messages, on_token=on_token)
        except ProviderError:
            log.exception("chat failed", extra={"mode": self.mode, "model": self.active.model})
            raise
        # Only keep turns that completed successfully.
        self.history.extend([user, Message("assistant", result.content)])
        return result
