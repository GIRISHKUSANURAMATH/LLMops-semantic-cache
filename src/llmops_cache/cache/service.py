"""Semantic cache engine: the only component that knows about embeddings, store and LLM."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from llmops_cache.cache.entry import CacheEntry
from llmops_cache.cache.identity import context_key, fingerprint, normalize_query
from llmops_cache.cache.vector_store import SearchHit, VectorStore
from llmops_cache.embeddings.base import EmbeddingProvider, Vector
from llmops_cache.observability.metrics import Metrics
from llmops_cache.providers.base import GenerationRequest, GenerationResult, LLMProvider

logger = logging.getLogger(__name__)

Source = Literal["llm", "semantic_cache", "coalesced"]


@dataclass(frozen=True, slots=True)
class CacheConfig:
    similarity_threshold: float = 0.90
    ttl_seconds: int = 3600
    top_k: int = 5
    max_temperature: float = 1.0
    log_prompts: bool = False


@dataclass(frozen=True, slots=True)
class ChatRequest:
    message: str
    model: str
    system_prompt: str | None = None
    temperature: float = 0.0
    max_tokens: int = 512
    namespace: str | None = None
    use_cache: bool = True
    ttl_seconds: int | None = None


@dataclass(frozen=True, slots=True)
class ChatResult:
    response: str
    source: Source
    cache_hit: bool
    similarity: float | None
    latency_ms: float
    model: str
    entry_id: str | None
    input_tokens: int
    output_tokens: int
    estimated_cost_saved_usd: float
    cache_bypass_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _Lookup:
    hit: SearchHit | None
    best_similarity: float | None


class SemanticCache:
    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        store: VectorStore,
        llm: LLMProvider,
        metrics: Metrics,
        config: CacheConfig,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.metrics = metrics
        self.config = config
        self._clock = clock
        self._inflight: dict[tuple[str, str], asyncio.Future[ChatResult]] = {}
        self.metrics.set_entries(len(store))

    # ------------------------------------------------------------------ public API
    async def chat(self, request: ChatRequest) -> ChatResult:
        started = time.perf_counter()
        normalized = normalize_query(request.message)
        key = self._context_key(request)
        log_ctx = self._log_context(request, normalized)
        logger.info("request.received", extra=log_ctx)

        bypass = self._bypass_reason(request)
        if bypass:
            return await self._call_llm_uncached(request, started, bypass, log_ctx)

        vector = await self._safe_embed(normalized, log_ctx)
        if vector is not None:
            lookup = self._safe_lookup(vector, key, self.config.similarity_threshold, log_ctx)
            if lookup.hit is not None:
                return self._serve_hit(lookup.hit, started, log_ctx)
            logger.info("cache.miss", extra={**log_ctx, "best_similarity": lookup.best_similarity})

        flight_key = (key, normalized)
        existing = self._inflight.get(flight_key)
        if existing is not None:
            joined = await self._join_flight(existing, started, log_ctx)
            if joined is not None:
                return joined
            # The leader was cancelled (e.g. client disconnected): do the work ourselves.
            return await self._generate_and_store(request, normalized, key, vector, started)

        future: asyncio.Future[ChatResult] = asyncio.get_running_loop().create_future()
        self._inflight[flight_key] = future
        try:
            result = await self._generate_and_store(request, normalized, key, vector, started)
        except asyncio.CancelledError:
            future.cancel()
            raise
        except Exception as exc:
            future.set_exception(exc)
            future.exception()  # mark retrieved so a leader without followers doesn't warn
            raise
        else:
            future.set_result(result)
            return result
        finally:
            self._inflight.pop(flight_key, None)

    async def search(
        self,
        request: ChatRequest,
        threshold: float | None = None,
        top_k: int | None = None,
    ) -> list[tuple[SearchHit, bool]]:
        """Inspect what the cache *would* match, without calling the LLM or counting hits."""
        normalized = normalize_query(request.message)
        vector = await self.embedder.embed_text(normalized)
        cutoff = self.config.similarity_threshold if threshold is None else threshold
        hits = self.store.search(vector, self._context_key(request), top_k or self.config.top_k)
        now = self._clock()
        return [(h, h.similarity >= cutoff and not h.entry.is_expired(now)) for h in hits]

    def clear(self) -> int:
        removed = self.store.clear()
        self.metrics.set_entries(len(self.store))
        logger.info("cache.cleared", extra={"removed": removed})
        return removed

    def delete(self, entry_id: str) -> bool:
        removed = self.store.delete(entry_id)
        self.metrics.set_entries(len(self.store))
        return removed

    def purge_expired(self) -> int:
        removed = self.store.purge_expired(self._clock())
        self.metrics.set_entries(len(self.store))
        if removed:
            logger.info("cache.purged", extra={"removed": removed})
        return removed

    # ----------------------------------------------------------------- internals
    def _context_key(self, request: ChatRequest) -> str:
        return context_key(
            provider=self.llm.cache_identity,
            model=request.model,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            namespace=request.namespace,
            embedding=self.embedder.name,
        )

    def _bypass_reason(self, request: ChatRequest) -> str | None:
        if not request.use_cache:
            return "disabled_by_request"
        if request.temperature > self.config.max_temperature:
            return "temperature_above_cache_limit"
        return None

    async def _safe_embed(self, normalized: str, log_ctx: dict[str, object]) -> Vector | None:
        try:
            return await self.embedder.embed_text(normalized)
        except Exception:
            logger.exception("embedding.failed", extra=log_ctx)
            return None

    def _safe_lookup(
        self, vector: Vector, key: str, threshold: float, log_ctx: dict[str, object]
    ) -> _Lookup:
        """Fail open: a broken cache must degrade to a plain LLM call, never to an error."""
        try:
            hits = self.store.search(vector, key, self.config.top_k)
        except Exception:
            logger.exception("cache.lookup_failed", extra=log_ctx)
            return _Lookup(None, None)
        now = self._clock()
        best = hits[0].similarity if hits else None
        for hit in hits:
            if hit.similarity < threshold:
                break
            if not hit.entry.is_expired(now):
                return _Lookup(hit, best)
        return _Lookup(None, best)

    def _serve_hit(self, hit: SearchHit, started: float, log_ctx: dict[str, object]) -> ChatResult:
        entry = hit.entry
        entry.record_hit(self._clock(), hit.similarity)
        saved = self.metrics.record_avoided(entry.input_tokens, entry.output_tokens)
        latency = _elapsed_ms(started)
        self.metrics.record("hit", latency)
        logger.info(
            "cache.hit",
            extra={
                **log_ctx,
                "similarity": round(hit.similarity, 4),
                "entry_id": entry.id,
                "latency_ms": latency,
            },
        )
        return ChatResult(
            response=entry.response,
            source="semantic_cache",
            cache_hit=True,
            similarity=round(hit.similarity, 4),
            latency_ms=latency,
            model=entry.model,
            entry_id=entry.id,
            input_tokens=entry.input_tokens,
            output_tokens=entry.output_tokens,
            estimated_cost_saved_usd=round(saved, 8),
        )

    async def _join_flight(
        self, future: asyncio.Future[ChatResult], started: float, log_ctx: dict[str, object]
    ) -> ChatResult | None:
        logger.info("cache.coalesced", extra=log_ctx)
        try:
            leader = await asyncio.shield(future)
        except asyncio.CancelledError:
            if future.cancelled():
                return None
            raise
        except Exception:
            self.metrics.record("error", _elapsed_ms(started))
            raise
        saved = self.metrics.record_avoided(leader.input_tokens, leader.output_tokens)
        latency = _elapsed_ms(started)
        self.metrics.record("coalesced", latency)
        return ChatResult(
            response=leader.response,
            source="coalesced",
            cache_hit=True,
            similarity=1.0,
            latency_ms=latency,
            model=leader.model,
            entry_id=leader.entry_id,
            input_tokens=leader.input_tokens,
            output_tokens=leader.output_tokens,
            estimated_cost_saved_usd=round(saved, 8),
        )

    async def _invoke_llm(self, request: ChatRequest) -> GenerationResult:
        self.metrics.record_llm_call()
        return await self.llm.generate(
            GenerationRequest(
                message=request.message,
                model=request.model,
                system_prompt=request.system_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
        )

    async def _generate_and_store(
        self,
        request: ChatRequest,
        normalized: str,
        key: str,
        vector: Vector | None,
        started: float,
    ) -> ChatResult:
        log_ctx = self._log_context(request, normalized)
        logger.info("llm.call", extra={**log_ctx, "provider": self.llm.name})
        try:
            generated = await self._invoke_llm(request)
        except Exception:
            self.metrics.record("error", _elapsed_ms(started))
            logger.exception("llm.failed", extra=log_ctx)
            raise

        entry_id = self._store(request, normalized, key, vector, generated, log_ctx)
        latency = _elapsed_ms(started)
        self.metrics.record("miss", latency)
        return ChatResult(
            response=generated.text,
            source="llm",
            cache_hit=False,
            similarity=None,
            latency_ms=latency,
            model=generated.model,
            entry_id=entry_id,
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
            estimated_cost_saved_usd=0.0,
        )

    def _store(
        self,
        request: ChatRequest,
        normalized: str,
        key: str,
        vector: Vector | None,
        generated: GenerationResult,
        log_ctx: dict[str, object],
    ) -> str | None:
        if vector is None or not generated.text.strip():
            return None
        now = self._clock()
        ttl = request.ttl_seconds or self.config.ttl_seconds
        entry = CacheEntry(
            id=uuid.uuid4().hex,
            context_key=key,
            query=request.message,
            normalized_query=normalized,
            response=generated.text,
            provider=self.llm.name,
            model=generated.model,
            embedding_model=self.embedder.name,
            created_at=now,
            expires_at=now + ttl,
            ttl_seconds=ttl,
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
        )
        try:
            self.store.add(entry, vector)
        except Exception:
            logger.exception("cache.insert_failed", extra=log_ctx)
            return None
        self.metrics.set_entries(len(self.store))
        logger.info("cache.insert", extra={**log_ctx, "entry_id": entry.id, "ttl_seconds": ttl})
        return entry.id

    async def _call_llm_uncached(
        self, request: ChatRequest, started: float, reason: str, log_ctx: dict[str, object]
    ) -> ChatResult:
        logger.info("cache.bypass", extra={**log_ctx, "reason": reason})
        try:
            generated = await self._invoke_llm(request)
        except Exception:
            self.metrics.record("error", _elapsed_ms(started))
            logger.exception("llm.failed", extra=log_ctx)
            raise
        latency = _elapsed_ms(started)
        self.metrics.record("bypass", latency)
        return ChatResult(
            response=generated.text,
            source="llm",
            cache_hit=False,
            similarity=None,
            latency_ms=latency,
            model=generated.model,
            entry_id=None,
            input_tokens=generated.input_tokens,
            output_tokens=generated.output_tokens,
            estimated_cost_saved_usd=0.0,
            cache_bypass_reason=reason,
        )

    def _log_context(self, request: ChatRequest, normalized: str) -> dict[str, object]:
        ctx: dict[str, object] = {"prompt_fp": fingerprint(normalized), "model": request.model}
        if self.config.log_prompts:
            ctx["prompt"] = request.message[:200]
        return ctx


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
