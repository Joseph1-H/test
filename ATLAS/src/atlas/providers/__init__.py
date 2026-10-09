"""Model providers. Add a new backend by subclassing ModelProvider and registering it below."""

from __future__ import annotations

from ..config import Settings
from .base import ChatResult, Message, ModelProvider, ProviderError
from .ollama import OllamaProvider
from .openai_compat import OpenAICompatProvider

__all__ = [
    "ChatResult",
    "Message",
    "ModelProvider",
    "OllamaProvider",
    "OpenAICompatProvider",
    "ProviderError",
    "make_cloud_provider",
    "make_local_provider",
]


def make_local_provider(settings: Settings, model: str | None = None) -> ModelProvider:
    return OllamaProvider(
        model=model or settings.local_model,
        host=settings.ollama_host,
        num_ctx=settings.num_ctx,
        temperature=settings.temperature,
        timeout=settings.request_timeout,
    )


def make_cloud_provider(settings: Settings, model: str | None = None) -> ModelProvider:
    if not settings.cloud_configured:
        raise ProviderError(
            "Cloud is not configured. Set ATLAS_CLOUD_BASE_URL, ATLAS_CLOUD_MODEL and "
            "ATLAS_CLOUD_API_KEY in your .env file."
        )
    return OpenAICompatProvider(
        model=model or settings.cloud_model,
        base_url=settings.cloud_base_url,
        api_key=settings.cloud_api_key,
        temperature=settings.temperature,
        timeout=settings.request_timeout,
    )
