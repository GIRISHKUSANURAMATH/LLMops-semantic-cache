"""Application factory and dependency wiring."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from llmops_cache import __version__
from llmops_cache.api.errors import RequestContextMiddleware, install_error_handlers
from llmops_cache.api.routes import public, v1
from llmops_cache.cache.service import CacheConfig, SemanticCache
from llmops_cache.cache.vector_store import InMemoryVectorStore
from llmops_cache.config import Settings, get_settings
from llmops_cache.embeddings import EmbeddingProvider, build_embedding_provider
from llmops_cache.observability.logging import configure_logging
from llmops_cache.observability.metrics import Metrics
from llmops_cache.providers import LLMProvider, build_llm_provider

logger = logging.getLogger(__name__)


def build_cache(
    settings: Settings,
    *,
    embedder: EmbeddingProvider | None = None,
    llm: LLMProvider | None = None,
    store: InMemoryVectorStore | None = None,
) -> SemanticCache:
    return SemanticCache(
        embedder=embedder if embedder is not None else build_embedding_provider(settings),
        store=store
        if store is not None
        else InMemoryVectorStore(max_entries=settings.cache_max_entries),
        llm=llm if llm is not None else build_llm_provider(settings),
        metrics=Metrics(
            cost_per_1k_input=settings.cost_per_1k_input_tokens,
            cost_per_1k_output=settings.cost_per_1k_output_tokens,
        ),
        config=CacheConfig(
            similarity_threshold=settings.cache_similarity_threshold,
            ttl_seconds=settings.cache_ttl_seconds,
            top_k=settings.cache_top_k,
            max_temperature=settings.cache_max_temperature,
            log_prompts=settings.log_prompts,
        ),
    )


async def _cleanup_loop(cache: SemanticCache, interval: int) -> None:
    while True:
        await asyncio.sleep(interval)
        try:
            cache.purge_expired()
        except Exception:
            logger.exception("cache.cleanup_failed")


async def _snapshot_loop(cache: SemanticCache, path: Path, interval: int) -> None:
    """Persist the index periodically so a crash or hard kill loses at most ``interval`` s."""
    store = cache.store
    assert isinstance(store, InMemoryVectorStore)  # noqa: S101 - checked by caller
    saved_version = store.version
    while True:
        await asyncio.sleep(interval)
        if store.version == saved_version:
            continue
        try:
            current = store.version
            store.save(path, cache.embedder.name)
            saved_version = current
            logger.info("cache.snapshot", extra={"entries": len(store)})
        except Exception:
            logger.exception("cache.snapshot_failed")


def create_app(settings: Settings | None = None, cache: SemanticCache | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)
    cache = cache if cache is not None else build_cache(settings)
    store_path = Path(settings.vector_store_path) if settings.vector_store_path else None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if store_path and isinstance(cache.store, InMemoryVectorStore):
            loaded = cache.store.load(store_path, cache.embedder.name, now=time.time())
            cache.metrics.set_entries(len(cache.store))
            logger.info("cache.loaded", extra={"entries": loaded, "path": str(store_path)})
        tasks: list[asyncio.Task[None]] = []
        if settings.cache_cleanup_interval_seconds > 0:
            tasks.append(
                asyncio.create_task(_cleanup_loop(cache, settings.cache_cleanup_interval_seconds))
            )
        persist = store_path is not None and isinstance(cache.store, InMemoryVectorStore)
        if persist and store_path and settings.cache_snapshot_interval_seconds > 0:
            tasks.append(
                asyncio.create_task(
                    _snapshot_loop(cache, store_path, settings.cache_snapshot_interval_seconds)
                )
            )
        logger.info(
            "app.started",
            extra={
                "llm_provider": cache.llm.name,
                "embedding_provider": cache.embedder.name,
                "threshold": cache.config.similarity_threshold,
            },
        )
        try:
            yield
        finally:
            for task in tasks:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            if store_path and isinstance(cache.store, InMemoryVectorStore):
                cache.store.save(store_path, cache.embedder.name)
                logger.info("cache.saved", extra={"entries": len(cache.store)})
            await cache.llm.aclose()
            await cache.embedder.aclose()

    app = FastAPI(
        title="LLMOPS — Semantic Caching Layer for LLM APIs",
        version=__version__,
        description=(
            "Semantic cache in front of an LLM provider. Similar prompts (by embedding "
            "cosine similarity, within the same model/system-prompt/parameter context) are "
            "served from cache. Cost figures are estimates."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.cache = cache

    if settings.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allow_origins,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Content-Type", "X-API-Key"],
        )
    app.add_middleware(RequestContextMiddleware, max_bytes=settings.max_request_bytes)
    install_error_handlers(app)
    app.include_router(public)
    app.include_router(v1)
    return app
