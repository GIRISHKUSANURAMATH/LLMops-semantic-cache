"""Behaviour of the semantic cache engine (the core of the project)."""

from __future__ import annotations

import asyncio
import io
import logging
import math

import pytest

from llmops_cache.cache.service import ChatRequest
from llmops_cache.cache.vector_store import InMemoryVectorStore, VectorStoreError
from llmops_cache.embeddings import EmbeddingError, Vector
from llmops_cache.observability.metrics import Metrics
from llmops_cache.providers import MockLLMProvider, ProviderError

from helpers import AngleEmbeddingProvider, FakeClock, make_cache


def req(message: str, **kw: object) -> ChatRequest:
    return ChatRequest(message=message, model="mock-1", **kw)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- hit / miss
async def test_first_query_calls_llm_and_is_a_miss(cache, llm: MockLLMProvider) -> None:
    result = await cache.chat(req("What is machine learning?"))
    assert result.cache_hit is False
    assert result.source == "llm"
    assert result.similarity is None
    assert result.entry_id is not None
    assert llm.calls == 1
    assert len(cache.store) == 1


async def test_similar_query_is_served_from_cache_without_llm_call(cache, llm) -> None:
    first = await cache.chat(req("What is machine learning?"))
    second = await cache.chat(req("Explain machine learning to me."))
    assert llm.calls == 1, "the LLM must NOT be called for a semantically similar query"
    assert second.cache_hit is True
    assert second.source == "semantic_cache"
    assert second.response == first.response
    assert second.similarity is not None and second.similarity >= 0.90
    assert second.estimated_cost_saved_usd > 0


async def test_different_query_calls_llm(cache, llm) -> None:
    await cache.chat(req("What is machine learning?"))
    result = await cache.chat(req("What is the capital of France?"))
    assert result.cache_hit is False
    assert llm.calls == 2
    assert len(cache.store) == 2


async def test_matching_is_not_plain_string_equality(cache, llm) -> None:
    await cache.chat(req("what is MACHINE learning"))
    hit = await cache.chat(req("  Can you explain Machine Learning?  "))
    assert hit.cache_hit is True
    assert llm.calls == 1


async def test_hit_updates_entry_hit_count_and_access_time(cache, clock: FakeClock) -> None:
    miss = await cache.chat(req("What is machine learning?"))
    clock.advance(5)
    await cache.chat(req("Explain machine learning"))
    entry = cache.store.get(miss.entry_id)
    assert entry is not None
    assert entry.hit_count == 1
    assert entry.last_accessed_at == clock.now
    assert entry.last_hit_similarity is not None


# --------------------------------------------------------------------------- threshold
@pytest.mark.parametrize(
    ("threshold", "expect_hit"),
    [(0.85, True), (0.90, True), (0.92, True), (0.93, False), (0.95, False)],
)
async def test_threshold_controls_borderline_similarity(
    threshold: float, expect_hit: bool, llm, clock, metrics
) -> None:
    embedder = AngleEmbeddingProvider()
    embedder.place("origin question", 1.0)
    embedder.place("borderline question", 0.92)  # exactly 0.92 cosine to the origin
    cache = make_cache(
        llm=llm, clock=clock, metrics=metrics, embedder=embedder, threshold=threshold
    )

    await cache.chat(req("origin question"))
    result = await cache.chat(req("borderline question"))

    assert result.cache_hit is expect_hit
    assert llm.calls == (1 if expect_hit else 2)
    if expect_hit:
        assert result.similarity == pytest.approx(0.92, abs=1e-4)


async def test_low_similarity_is_a_miss_even_with_permissive_threshold(llm, clock, metrics):
    embedder = AngleEmbeddingProvider()
    embedder.place("a", 1.0)
    embedder.place("b", 0.30)
    cache = make_cache(llm=llm, clock=clock, metrics=metrics, embedder=embedder, threshold=0.5)
    await cache.chat(req("a"))
    assert (await cache.chat(req("b"))).cache_hit is False
    assert llm.calls == 2


# --------------------------------------------------------------------------- TTL
async def test_expired_entry_never_produces_a_hit(cache, llm, clock: FakeClock) -> None:
    await cache.chat(req("What is machine learning?"))
    clock.advance(61)  # ttl is 60s in make_cache
    result = await cache.chat(req("What is machine learning?"))
    assert result.cache_hit is False
    assert llm.calls == 2


async def test_entry_is_valid_until_just_before_expiry(cache, llm, clock) -> None:
    await cache.chat(req("What is machine learning?"))
    clock.advance(59.9)
    assert (await cache.chat(req("What is machine learning?"))).cache_hit is True
    clock.advance(0.1)  # now == expires_at -> expired
    assert (await cache.chat(req("What is machine learning?"))).cache_hit is False


async def test_per_request_ttl_override(cache, llm, clock) -> None:
    await cache.chat(req("What is machine learning?", ttl_seconds=5))
    clock.advance(6)
    assert (await cache.chat(req("What is machine learning?"))).cache_hit is False


async def test_valid_lower_ranked_match_is_used_when_best_match_expired(llm, clock, metrics):
    # Angles: A=0deg, B=40deg, probe=18deg -> probe~A 0.951, probe~B 0.927, A~B 0.766.
    embedder = AngleEmbeddingProvider()
    embedder.place("A", 1.0)
    embedder.place("B", math.cos(math.radians(40)))
    embedder.place("probe", math.cos(math.radians(18)))
    cache = make_cache(
        llm=llm, clock=clock, metrics=metrics, embedder=embedder, threshold=0.9, ttl=60
    )
    entry_a = await cache.chat(req("A"))
    clock.advance(30)
    entry_b = await cache.chat(req("B"))
    assert entry_b.cache_hit is False  # A and B are too far apart to share an answer

    clock.advance(40)  # A has expired, B has not
    result = await cache.chat(req("probe"))
    assert result.cache_hit is True
    assert result.entry_id == entry_b.entry_id != entry_a.entry_id
    assert result.similarity == pytest.approx(math.cos(math.radians(22)), abs=1e-3)


async def test_purge_expired_removes_only_stale_entries(cache, clock) -> None:
    await cache.chat(req("What is machine learning?"))
    clock.advance(30)
    await cache.chat(req("What is the capital of France?"))
    clock.advance(31)
    assert cache.purge_expired() == 1
    assert len(cache.store) == 1
    assert cache.metrics.snapshot().entries_count == 1


# --------------------------------------------------------------------------- cache identity
@pytest.mark.parametrize(
    "variant",
    [
        {"model": "other-model"},
        {"system_prompt": "You are a pirate."},
        {"temperature": 0.7},
        {"max_tokens": 64},
        {"namespace": "tenant-b"},
    ],
)
async def test_different_generation_context_never_shares_cache(cache, llm, variant) -> None:
    await cache.chat(req("What is machine learning?"))
    params = {"message": "What is machine learning?", "model": "mock-1", **variant}
    result = await cache.chat(ChatRequest(**params))  # type: ignore[arg-type]
    assert result.cache_hit is False
    assert llm.calls == 2


async def test_cache_bypass_when_requested(cache, llm) -> None:
    await cache.chat(req("What is machine learning?"))
    result = await cache.chat(req("What is machine learning?", use_cache=False))
    assert result.cache_hit is False
    assert result.cache_bypass_reason == "disabled_by_request"
    assert llm.calls == 2


async def test_high_temperature_bypasses_cache(llm, clock, metrics) -> None:
    cache = make_cache(llm=llm, clock=clock, metrics=metrics)
    cache.config = type(cache.config)(max_temperature=0.5)
    first = await cache.chat(req("What is machine learning?", temperature=0.9))
    assert first.cache_bypass_reason == "temperature_above_cache_limit"
    assert len(cache.store) == 0


# --------------------------------------------------------------------------- clear / delete
async def test_clear_removes_all_entries_and_next_request_misses(cache, llm) -> None:
    await cache.chat(req("What is machine learning?"))
    await cache.chat(req("What is the capital of France?"))
    assert cache.clear() == 2
    assert len(cache.store) == 0
    assert (await cache.chat(req("What is machine learning?"))).cache_hit is False
    assert llm.calls == 3


async def test_delete_single_entry(cache) -> None:
    result = await cache.chat(req("What is machine learning?"))
    assert cache.delete(result.entry_id) is True
    assert cache.delete(result.entry_id) is False
    assert len(cache.store) == 0


# --------------------------------------------------------------------------- failures
async def test_provider_failure_propagates_and_nothing_is_cached(cache, llm) -> None:
    llm.fail_with = ProviderError("upstream down")
    with pytest.raises(ProviderError):
        await cache.chat(req("What is machine learning?"))
    assert len(cache.store) == 0
    assert cache.metrics.snapshot().errors == 1

    llm.fail_with = None
    result = await cache.chat(req("What is machine learning?"))
    assert result.cache_hit is False  # the failed call left no poisoned entry


class _BrokenEmbedder(AngleEmbeddingProvider):
    async def embed_batch(self, texts: list[str]) -> list[Vector]:
        raise EmbeddingError("model crashed")


async def test_embedding_failure_fails_open_to_llm(llm, clock, metrics) -> None:
    cache = make_cache(llm=llm, clock=clock, metrics=metrics, embedder=_BrokenEmbedder())
    result = await cache.chat(req("What is machine learning?"))
    assert result.source == "llm"
    assert result.entry_id is None  # cannot be cached without a vector
    assert llm.calls == 1


class _BrokenSearchStore(InMemoryVectorStore):
    def search(self, vector, context_key, top_k):  # type: ignore[no-untyped-def]
        raise VectorStoreError("index corrupted")


class _BrokenAddStore(InMemoryVectorStore):
    def add(self, entry, vector):  # type: ignore[no-untyped-def]
        raise VectorStoreError("disk full")


async def test_vector_store_search_failure_fails_open(llm, clock, metrics) -> None:
    cache = make_cache(llm=llm, clock=clock, metrics=metrics, store=_BrokenSearchStore())
    result = await cache.chat(req("What is machine learning?"))
    assert result.source == "llm"
    assert result.response


async def test_vector_store_insert_failure_still_returns_response(llm, clock, metrics) -> None:
    cache = make_cache(llm=llm, clock=clock, metrics=metrics, store=_BrokenAddStore())
    result = await cache.chat(req("What is machine learning?"))
    assert result.response
    assert result.entry_id is None


# --------------------------------------------------------------------------- concurrency
async def test_concurrent_identical_requests_make_one_llm_call(clock, metrics) -> None:
    slow = MockLLMProvider(latency_ms=50)
    cache = make_cache(llm=slow, clock=clock, metrics=metrics)
    results = await asyncio.gather(
        *(cache.chat(req("What is machine learning?")) for _ in range(10))
    )
    assert slow.calls == 1
    assert len({r.response for r in results}) == 1
    assert sum(r.source == "llm" for r in results) == 1
    assert sum(r.source == "coalesced" for r in results) == 9
    snap = cache.metrics.snapshot()
    assert snap.llm_calls == 1
    assert snap.llm_calls_avoided == 9


async def test_concurrent_followers_receive_leader_failure(clock, metrics) -> None:
    slow = MockLLMProvider(latency_ms=30)
    slow.fail_with = ProviderError("boom")
    cache = make_cache(llm=slow, clock=clock, metrics=metrics)
    results = await asyncio.gather(
        *(cache.chat(req("same")) for _ in range(3)), return_exceptions=True
    )
    assert all(isinstance(r, ProviderError) for r in results)
    assert slow.calls == 1
    assert len(cache._inflight) == 0


async def test_cancelled_leader_does_not_strand_followers(clock, metrics) -> None:
    slow = MockLLMProvider(latency_ms=100)
    cache = make_cache(llm=slow, clock=clock, metrics=metrics)
    leader = asyncio.create_task(cache.chat(req("same")))
    await asyncio.sleep(0.01)
    follower = asyncio.create_task(cache.chat(req("same")))
    await asyncio.sleep(0.01)
    leader.cancel()
    result = await follower
    assert result.source == "llm"
    assert slow.calls == 2


# --------------------------------------------------------------------------- metrics
async def test_metrics_are_accurate(cache, llm) -> None:
    await cache.chat(req("What is machine learning?"))  # miss
    await cache.chat(req("Explain machine learning to me"))  # hit
    await cache.chat(req("machine learning"))  # hit
    await cache.chat(req("What is the capital of France?"))  # miss
    await cache.chat(req("bypass me", use_cache=False))  # bypass
    snap = cache.metrics.snapshot()
    assert snap.requests_total == 5
    assert snap.cache_hits == 2
    assert snap.cache_misses == 2
    assert snap.cache_bypassed == 1
    assert snap.hit_rate == 0.5
    assert snap.llm_calls == llm.calls == 3
    assert snap.llm_calls_avoided == 2
    assert snap.entries_count == 2
    assert snap.estimated_tokens_avoided > 0
    assert snap.average_hit_latency_ms is not None
    assert snap.average_miss_latency_ms is not None


async def test_cost_saved_uses_configured_prices(llm, clock) -> None:
    metrics = Metrics(cost_per_1k_input=1.0, cost_per_1k_output=2.0)
    cache = make_cache(llm=llm, clock=clock, metrics=metrics)
    miss = await cache.chat(req("What is machine learning?"))
    hit = await cache.chat(req("machine learning"))
    expected = miss.input_tokens / 1000 * 1.0 + miss.output_tokens / 1000 * 2.0
    assert hit.estimated_cost_saved_usd == pytest.approx(expected)
    assert metrics.snapshot().estimated_cost_saved_usd == pytest.approx(expected, abs=1e-6)


# --------------------------------------------------------------------------- logging
async def test_prompts_are_not_logged_by_default(cache) -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("llmops_cache")
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        await cache.chat(req("my secret medical question"))
        await cache.chat(req("my secret medical question"))
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)
    output = stream.getvalue()
    assert "cache.hit" in output
    assert "secret medical" not in output
