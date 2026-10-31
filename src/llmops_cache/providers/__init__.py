"""LLM providers."""

from __future__ import annotations

from llmops_cache.config import Settings
from llmops_cache.providers.base import (
    GenerationRequest,
    GenerationResult,
    LLMProvider,
    ProviderError,
)
from llmops_cache.providers.mock import MockLLMProvider

__all__ = [
    "GenerationRequest",
    "GenerationResult",
    "LLMProvider",
    "MockLLMProvider",
    "ProviderError",
    "build_llm_provider",
]


def build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "mock":
        return MockLLMProvider(latency_ms=settings.mock_llm_latency_ms)
    from llmops_cache.providers.openai_compat import OpenAICompatibleProvider

    assert settings.openai_api_key is not None  # noqa: S101 - enforced by Settings validator
    return OpenAICompatibleProvider(
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.openai_base_url,
        timeout=settings.llm_timeout_seconds,
    )
