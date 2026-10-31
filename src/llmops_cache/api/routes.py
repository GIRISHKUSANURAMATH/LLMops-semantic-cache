"""HTTP routes. Thin: translate HTTP <-> service objects, no caching logic here."""

from __future__ import annotations

import hmac
import time

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import APIKeyHeader
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from llmops_cache import __version__
from llmops_cache.api.schemas import (
    CacheSearchBody,
    CacheSearchMatch,
    CacheSearchResponse,
    CacheStats,
    ChatRequestBody,
    ChatResponseBody,
    ClearResponse,
    ErrorBody,
    HealthResponse,
    MetricsResponse,
    Usage,
)
from llmops_cache.cache.service import ChatRequest, SemanticCache
from llmops_cache.config import Settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_cache(request: Request) -> SemanticCache:
    cache: SemanticCache = request.app.state.cache
    return cache


def get_settings_dep(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def require_api_key(
    settings: Settings = Depends(get_settings_dep),
    api_key: str | None = Depends(_api_key_header),
) -> None:
    """Enforced only when API_KEYS is configured. Constant-time comparison."""
    if not settings.api_keys:
        return
    supplied = (api_key or "").encode()
    if not any(
        hmac.compare_digest(supplied, k.get_secret_value().encode()) for k in settings.api_keys
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


_ERRORS: dict[int | str, dict[str, object]] = {
    401: {"model": ErrorBody},
    413: {"model": ErrorBody},
    422: {"model": ErrorBody},
    500: {"model": ErrorBody},
    502: {"model": ErrorBody},
    503: {"model": ErrorBody},
}

public = APIRouter()
v1 = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)], responses=_ERRORS)


@public.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {"name": "llmops-semantic-cache", "version": __version__, "docs": "/docs"}


@public.get("/health", response_model=HealthResponse, tags=["system"])
def health(cache: SemanticCache = Depends(get_cache)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        llm_provider=cache.llm.name,
        embedding_provider=cache.embedder.name,
        entries_count=len(cache.store),
    )


def _to_service_request(
    body: ChatRequestBody | CacheSearchBody, settings: Settings, max_chars: int
) -> ChatRequest:
    if len(body.message) > max_chars:
        raise HTTPException(
            status_code=413,
            detail=f"message exceeds {max_chars} characters",
        )
    model = settings.llm_default_model if body.model == "default" else body.model
    use_cache = body.use_cache if isinstance(body, ChatRequestBody) else True
    ttl = body.ttl_seconds if isinstance(body, ChatRequestBody) else None
    return ChatRequest(
        message=body.message,
        model=model,
        system_prompt=body.system_prompt,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        namespace=body.namespace,
        use_cache=use_cache,
        ttl_seconds=ttl,
    )


@v1.post("/chat", response_model=ChatResponseBody, tags=["chat"])
async def chat(
    body: ChatRequestBody,
    cache: SemanticCache = Depends(get_cache),
    settings: Settings = Depends(get_settings_dep),
) -> ChatResponseBody:
    result = await cache.chat(_to_service_request(body, settings, settings.max_message_chars))
    return ChatResponseBody(
        response=result.response,
        cache_hit=result.cache_hit,
        source=result.source,
        similarity=result.similarity,
        latency_ms=result.latency_ms,
        model=result.model,
        cache_entry_id=result.entry_id,
        usage=Usage(input_tokens=result.input_tokens, output_tokens=result.output_tokens),
        estimated_cost_saved_usd=result.estimated_cost_saved_usd,
        cache_bypass_reason=result.cache_bypass_reason,
    )


@v1.post("/cache/search", response_model=CacheSearchResponse, tags=["cache"])
async def cache_search(
    body: CacheSearchBody,
    cache: SemanticCache = Depends(get_cache),
    settings: Settings = Depends(get_settings_dep),
) -> CacheSearchResponse:
    request = _to_service_request(body, settings, settings.max_message_chars)
    threshold = body.threshold or cache.config.similarity_threshold
    results = await cache.search(request, threshold=threshold, top_k=body.top_k)
    now = time.time()
    return CacheSearchResponse(
        threshold=threshold,
        matches=[
            CacheSearchMatch(
                entry_id=hit.entry.id,
                query=hit.entry.query,
                similarity=round(hit.similarity, 4),
                would_hit=would_hit,
                expired=hit.entry.is_expired(now),
                hit_count=hit.entry.hit_count,
                created_at=hit.entry.created_at,
                expires_at=hit.entry.expires_at,
            )
            for hit, would_hit in results
        ],
    )


@v1.delete("/cache", response_model=ClearResponse, tags=["cache"])
def clear_cache(cache: SemanticCache = Depends(get_cache)) -> ClearResponse:
    return ClearResponse(removed=cache.clear())


@v1.delete("/cache/expired", response_model=ClearResponse, tags=["cache"])
def purge_expired(cache: SemanticCache = Depends(get_cache)) -> ClearResponse:
    return ClearResponse(removed=cache.purge_expired())


@v1.delete("/cache/{entry_id}", response_model=ClearResponse, tags=["cache"])
def delete_entry(entry_id: str, cache: SemanticCache = Depends(get_cache)) -> ClearResponse:
    if not cache.delete(entry_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="cache entry not found")
    return ClearResponse(removed=1)


@v1.get("/cache/stats", response_model=CacheStats, tags=["cache"])
def cache_stats(
    cache: SemanticCache = Depends(get_cache),
    settings: Settings = Depends(get_settings_dep),
) -> CacheStats:
    now = time.time()
    entries = cache.store.entries()
    return CacheStats(
        entries_count=len(entries),
        max_entries=settings.cache_max_entries,
        expired_entries=sum(1 for e in entries if e.is_expired(now)),
        total_hits_served=sum(e.hit_count for e in entries),
        similarity_threshold=cache.config.similarity_threshold,
        ttl_seconds=cache.config.ttl_seconds,
        embedding_provider=cache.embedder.name,
        embedding_dimensions=cache.embedder.dimensions,
        llm_provider=cache.llm.name,
    )


@v1.get("/metrics", response_model=MetricsResponse, tags=["observability"])
def metrics(cache: SemanticCache = Depends(get_cache)) -> MetricsResponse:
    cache.metrics.set_entries(len(cache.store))
    snap = cache.metrics.snapshot()
    return MetricsResponse(**vars(snap))


@public.get("/metrics", include_in_schema=False, dependencies=[Depends(require_api_key)])
def prometheus_metrics(cache: SemanticCache = Depends(get_cache)) -> Response:
    cache.metrics.set_entries(len(cache.store))
    return Response(generate_latest(cache.metrics.registry), media_type=CONTENT_TYPE_LATEST)
