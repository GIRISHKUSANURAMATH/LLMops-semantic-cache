"""Test helpers shared across the unit, integration and e2e suites."""

from __future__ import annotations

import math

import numpy as np
from fastapi import FastAPI

from llmops_cache.cache.identity import normalize_query
from llmops_cache.cache.service import CacheConfig, SemanticCache
from llmops_cache.cache.vector_store import InMemoryVectorStore
from llmops_cache.config import Settings
from llmops_cache.embeddings import EmbeddingProvider, HashingEmbeddingProvider, Vector
from llmops_cache.observability.metrics import Metrics
from llmops_cache.providers import MockLLMProvider


class FakeClock:
    def __init__(self, start: float = 1_700_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class AngleEmbeddingProvider(EmbeddingProvider):
    """Maps registered texts to 2-D unit vectors at a given angle.

    cos(angle_a - angle_b) is the exact similarity, so tests can place two queries at
    *precisely* 0.92 similarity and assert threshold behaviour without approximations.
    """

    name = "angle-test"
    dimensions = 2

    def __init__(self) -> None:
        self._angles: dict[str, float] = {}

    def place(self, text: str, similarity_to_origin: float) -> None:
        self._angles[normalize_query(text)] = math.acos(similarity_to_origin)

    async def embed_batch(self, texts: list[str]) -> list[Vector]:
        out = []
        for text in texts:
            angle = self._angles.get(text, math.pi / 2)
            out.append(np.array([math.cos(angle), math.sin(angle)], dtype=np.float32))
        return out


def make_cache(
    *,
    llm: MockLLMProvider,
    clock: FakeClock,
    metrics: Metrics,
    embedder: EmbeddingProvider | None = None,
    threshold: float = 0.90,
    ttl: int = 60,
    store: InMemoryVectorStore | None = None,
) -> SemanticCache:
    return SemanticCache(
        embedder=embedder if embedder is not None else HashingEmbeddingProvider(),
        store=store if store is not None else InMemoryVectorStore(max_entries=100),
        llm=llm,
        metrics=metrics,
        config=CacheConfig(similarity_threshold=threshold, ttl_seconds=ttl),
        clock=clock,
    )


def make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "app_env": "test",
        "mock_llm_latency_ms": 0,
        "log_level": "WARNING",
        "cache_cleanup_interval_seconds": 0,
        "vector_store_path": "",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def mock_llm_of(app: FastAPI) -> MockLLMProvider:
    provider = app.state.cache.llm
    assert isinstance(provider, MockLLMProvider)
    return provider
