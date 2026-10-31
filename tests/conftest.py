"""Shared fixtures. Nothing here touches the network or needs an API key."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
from fastapi import FastAPI

from llmops_cache.cache.service import SemanticCache
from llmops_cache.config import Settings
from llmops_cache.main import create_app
from llmops_cache.observability.metrics import Metrics
from llmops_cache.providers import MockLLMProvider

from helpers import FakeClock, make_cache, make_settings


@pytest.fixture(autouse=True)
def _isolated_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """Never read a developer's real .env or OPENAI_API_KEY during tests."""
    for var in (
        "OPENAI_API_KEY",
        "LLM_PROVIDER",
        "EMBEDDING_PROVIDER",
        "API_KEYS",
        "CORS_ALLOW_ORIGINS",
        "VECTOR_STORE_PATH",
        "APP_ENV",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))
    yield


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def llm() -> MockLLMProvider:
    return MockLLMProvider(latency_ms=0)


@pytest.fixture
def metrics() -> Metrics:
    return Metrics(cost_per_1k_input=0.001, cost_per_1k_output=0.002)


@pytest.fixture
def cache(llm: MockLLMProvider, clock: FakeClock, metrics: Metrics) -> SemanticCache:
    return make_cache(llm=llm, clock=clock, metrics=metrics)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
async def app(settings: Settings) -> AsyncIterator[FastAPI]:
    application = create_app(settings)
    async with application.router.lifespan_context(application):
        yield application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http
