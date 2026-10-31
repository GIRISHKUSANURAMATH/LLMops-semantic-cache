"""HTTP API behaviour through the real FastAPI app (in-process ASGI, no network)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from llmops_cache.main import create_app
from llmops_cache.providers import ProviderError

from helpers import make_settings, mock_llm_of

pytestmark = pytest.mark.integration

CHAT_FIELDS = {
    "response",
    "cache_hit",
    "source",
    "similarity",
    "latency_ms",
    "model",
    "cache_entry_id",
    "usage",
    "estimated_cost_saved_usd",
    "cache_bypass_reason",
}


async def _client_for(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


# --------------------------------------------------------------------------- system
async def test_health_and_root(client: httpx.AsyncClient) -> None:
    health = await client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] == "ok"
    assert body["llm_provider"] == "mock"
    assert body["embedding_provider"].startswith("hashing")
    assert (await client.get("/")).json()["docs"] == "/docs"
    assert "x-request-id" in health.headers


async def test_openapi_docs_and_redoc_are_served(client: httpx.AsyncClient) -> None:
    spec = (await client.get("/openapi.json")).json()
    for path in ("/v1/chat", "/v1/cache/search", "/v1/cache", "/v1/cache/stats", "/v1/metrics"):
        assert path in spec["paths"], path
    assert (await client.get("/docs")).status_code == 200
    assert (await client.get("/redoc")).status_code == 200


# --------------------------------------------------------------------------- chat
async def test_chat_miss_then_semantic_hit(client: httpx.AsyncClient, app: FastAPI) -> None:
    miss = await client.post("/v1/chat", json={"message": "What is machine learning?"})
    assert miss.status_code == 200
    miss_body = miss.json()
    assert set(miss_body) == CHAT_FIELDS
    assert miss_body["cache_hit"] is False
    assert miss_body["source"] == "llm"
    assert miss_body["similarity"] is None
    assert miss_body["model"] == "mock-1"  # "default" resolved to LLM_DEFAULT_MODEL

    hit = await client.post("/v1/chat", json={"message": "Explain machine learning to me."})
    hit_body = hit.json()
    assert hit_body["cache_hit"] is True
    assert hit_body["source"] == "semantic_cache"
    assert hit_body["similarity"] >= 0.9
    assert hit_body["response"] == miss_body["response"]
    assert hit_body["cache_entry_id"] == miss_body["cache_entry_id"]
    assert mock_llm_of(app).calls == 1

    other = await client.post("/v1/chat", json={"message": "What is the capital of France?"})
    assert other.json()["cache_hit"] is False
    assert mock_llm_of(app).calls == 2


async def test_chat_response_does_not_leak_other_users_prompt(client) -> None:
    await client.post("/v1/chat", json={"message": "machine learning for my secret project"})
    hit = await client.post("/v1/chat", json={"message": "machine learning secret project"})
    assert "query" not in hit.json()


async def test_namespace_isolates_tenants(client, app) -> None:
    await client.post("/v1/chat", json={"message": "pricing plans", "namespace": "tenant-a"})
    other = await client.post(
        "/v1/chat", json={"message": "pricing plans", "namespace": "tenant-b"}
    )
    assert other.json()["cache_hit"] is False
    assert mock_llm_of(app).calls == 2


# --------------------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": ""},
        {"message": "   "},
        {"message": 123},
        {"message": "hi", "temperature": 5},
        {"message": "hi", "max_tokens": 0},
        {"message": "hi", "namespace": "../../etc"},
        {"message": "hi", "unknown_field": True},
    ],
)
async def test_invalid_input_returns_422(client: httpx.AsyncClient, payload: dict) -> None:
    response = await client.post("/v1/chat", json=payload)
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


async def test_malformed_json_returns_422(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/v1/chat", content=b'{"message": ', headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
    assert "Traceback" not in response.text


async def test_oversized_body_rejected(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/chat", json={"message": "x" * 100_000})
    assert response.status_code == 413
    assert response.json()["error"] == "payload_too_large"


async def test_oversized_message_rejected(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/chat", json={"message": "x" * 9000})
    assert response.status_code == 413


async def test_unknown_route_is_json_404(client: httpx.AsyncClient) -> None:
    response = await client.get("/v1/nope")
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


# --------------------------------------------------------------------------- failures
async def test_provider_failure_returns_503_without_internals(client, app) -> None:
    mock_llm_of(app).fail_with = ProviderError("secret upstream detail at 10.0.0.5")
    response = await client.post("/v1/chat", json={"message": "hello there"})
    assert response.status_code == 503
    assert response.json()["error"] == "provider_unavailable"
    assert "10.0.0.5" not in response.text
    assert "Traceback" not in response.text


async def test_non_retryable_provider_failure_returns_502(client, app) -> None:
    mock_llm_of(app).fail_with = ProviderError("bad request", retryable=False)
    assert (await client.post("/v1/chat", json={"message": "hi"})).status_code == 502


async def test_unexpected_error_returns_generic_500(client, app) -> None:
    async def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("database password=hunter2")

    app.state.cache.llm.generate = explode
    response = await client.post("/v1/chat", json={"message": "hi"})
    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "internal_error"
    assert body["message"] == "Internal server error."
    assert isinstance(body["request_id"], str)
    assert "hunter2" not in response.text


# --------------------------------------------------------------------------- cache endpoints
async def test_cache_search_reports_matches_without_calling_llm(client, app) -> None:
    await client.post("/v1/chat", json={"message": "What is machine learning?"})
    calls = mock_llm_of(app).calls
    response = await client.post("/v1/cache/search", json={"message": "explain machine learning"})
    body = response.json()
    assert response.status_code == 200
    assert body["threshold"] == 0.9
    assert body["matches"][0]["would_hit"] is True
    assert body["matches"][0]["query"] == "What is machine learning?"
    assert mock_llm_of(app).calls == calls

    strict = await client.post(
        "/v1/cache/search", json={"message": "machine learning basics", "threshold": 0.99}
    )
    assert all(not m["would_hit"] for m in strict.json()["matches"])


async def test_clear_cache_endpoint(client, app) -> None:
    await client.post("/v1/chat", json={"message": "What is machine learning?"})
    await client.post("/v1/chat", json={"message": "What is the capital of France?"})
    cleared = await client.delete("/v1/cache")
    assert cleared.json() == {"removed": 2}
    assert (await client.get("/v1/cache/stats")).json()["entries_count"] == 0
    again = await client.post("/v1/chat", json={"message": "What is machine learning?"})
    assert again.json()["cache_hit"] is False


async def test_delete_entry_and_404(client) -> None:
    created = await client.post("/v1/chat", json={"message": "What is machine learning?"})
    entry_id = created.json()["cache_entry_id"]
    assert (await client.delete(f"/v1/cache/{entry_id}")).status_code == 200
    assert (await client.delete(f"/v1/cache/{entry_id}")).status_code == 404


async def test_stats_and_metrics_are_accurate(client) -> None:
    for message in [
        "What is machine learning?",
        "Explain machine learning",
        "machine learning?",
        "Capital of France",
    ]:
        await client.post("/v1/chat", json={"message": message})
    stats = (await client.get("/v1/cache/stats")).json()
    assert stats["entries_count"] == 2
    assert stats["total_hits_served"] == 2
    assert stats["similarity_threshold"] == 0.9

    metrics = (await client.get("/v1/metrics")).json()
    assert metrics["requests_total"] == 4
    assert metrics["cache_hits"] == 2
    assert metrics["cache_misses"] == 2
    assert metrics["hit_rate"] == 0.5
    assert metrics["llm_calls"] == 2
    assert metrics["llm_calls_avoided"] == 2
    assert metrics["estimated_cost_saved_usd"] > 0
    assert "estimate" in metrics["note"]


async def test_prometheus_endpoint(client) -> None:
    await client.post("/v1/chat", json={"message": "hello"})
    text = (await client.get("/metrics")).text
    assert 'llmops_requests_total{outcome="miss"} 1.0' in text
    assert "llmops_cache_entries 1.0" in text


# --------------------------------------------------------------------------- security
async def test_api_key_enforced_when_configured() -> None:
    app = create_app(make_settings(api_keys=["s3cret"]))
    async for client in _client_for(app):
        assert (await client.get("/health")).status_code == 200  # liveness stays open
        denied = await client.post("/v1/chat", json={"message": "hi"})
        assert denied.status_code == 401
        wrong = await client.post("/v1/chat", json={"message": "hi"}, headers={"X-API-Key": "nope"})
        assert wrong.status_code == 401
        ok = await client.post("/v1/chat", json={"message": "hi"}, headers={"X-API-Key": "s3cret"})
        assert ok.status_code == 200
        assert (await client.get("/metrics")).status_code == 401


async def test_cors_disabled_by_default(client) -> None:
    response = await client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in response.headers


async def test_cors_allows_only_configured_origins() -> None:
    app = create_app(make_settings(cors_allow_origins=["https://app.example.com"]))
    async for client in _client_for(app):
        good = await client.get("/health", headers={"Origin": "https://app.example.com"})
        assert good.headers["access-control-allow-origin"] == "https://app.example.com"
        bad = await client.get("/health", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in bad.headers


# --------------------------------------------------------------------------- persistence
async def test_cache_survives_restart_when_vector_store_path_set(tmp_path: Path) -> None:
    settings = make_settings(vector_store_path=str(tmp_path / "store"))
    first = create_app(settings)
    async for client in _client_for(first):
        await client.post("/v1/chat", json={"message": "What is machine learning?"})
    assert (tmp_path / "store" / "entries.json").exists()

    second = create_app(settings)
    async for client in _client_for(second):
        hit = await client.post("/v1/chat", json={"message": "Explain machine learning"})
        assert hit.json()["cache_hit"] is True
        assert mock_llm_of(second).calls == 0
