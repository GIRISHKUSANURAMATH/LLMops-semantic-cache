"""Embedding providers."""

from __future__ import annotations

from llmops_cache.config import Settings
from llmops_cache.embeddings.base import EmbeddingError, EmbeddingProvider, Vector
from llmops_cache.embeddings.hashing import HashingEmbeddingProvider

__all__ = [
    "EmbeddingError",
    "EmbeddingProvider",
    "HashingEmbeddingProvider",
    "Vector",
    "build_embedding_provider",
]


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "hashing":
        return HashingEmbeddingProvider(dimensions=settings.embedding_dimensions)
    if settings.embedding_provider == "sentence-transformers":
        from llmops_cache.embeddings.sentence_transformer import (
            SentenceTransformerEmbeddingProvider,
        )

        return SentenceTransformerEmbeddingProvider(settings.embedding_model)
    from llmops_cache.embeddings.openai import OpenAIEmbeddingProvider

    assert settings.openai_api_key is not None  # noqa: S101 - enforced by Settings validator
    return OpenAIEmbeddingProvider(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.openai_embedding_model,
        base_url=settings.openai_base_url,
        timeout=settings.llm_timeout_seconds,
    )
