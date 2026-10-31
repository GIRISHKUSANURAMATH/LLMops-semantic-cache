from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from llmops_cache.config import Settings
from llmops_cache.providers import GenerationRequest, MockLLMProvider, ProviderError
from llmops_cache.providers.openai_compat import OpenAICompatibleProvider

REQUEST = GenerationRequest(
    message="hi", model="gpt-x", system_prompt="be brief", temperature=0.2, max_tokens=20
)


# --------------------------------------------------------------------------- mock provider
async def test_mock_provider_is_deterministic_and_counts_calls() -> None:
    llm = MockLLMProvider(latency_ms=0)
    a = await llm.generate(REQUEST)
    b = await llm.generate(REQUEST)
    assert a == b
    assert llm.calls == 2
    assert a.input_tokens > 0 and a.output_tokens > 0


async def test_mock_provider_can_simulate_failure() -> None:
    llm = MockLLMProvider(latency_ms=0)
    llm.fail_with = ProviderError("down")
    with pytest.raises(ProviderError):
        await llm.generate(REQUEST)


# --------------------------------------------------------------------------- openai-compatible
def _provider(handler) -> OpenAICompatibleProvider:  # type: ignore[no-untyped-def]
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://x/v1")
    return OpenAICompatibleProvider(api_key="sk-test", base_url="https://x/v1", client=client)


async def test_openai_provider_sends_expected_payload_and_parses_usage() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "gpt-x",
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            },
        )

    result = await _provider(handler).generate(REQUEST)
    assert result.text == "hello"
    assert (result.input_tokens, result.output_tokens) == (7, 3)
    assert seen["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    assert seen["temperature"] == 0.2 and seen["max_tokens"] == 20


@pytest.mark.parametrize(
    ("status", "retryable"), [(500, True), (503, True), (429, True), (401, False), (400, False)]
)
async def test_openai_provider_maps_http_errors(status: int, retryable: bool) -> None:
    provider = _provider(lambda request: httpx.Response(status, json={"error": "x"}))
    with pytest.raises(ProviderError) as info:
        await provider.generate(REQUEST)
    assert info.value.retryable is retryable
    assert "sk-test" not in str(info.value)


async def test_openai_provider_handles_timeout_and_bad_payload() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ProviderError, match="timed out"):
        await _provider(timeout).generate(REQUEST)
    with pytest.raises(ProviderError, match="unexpected payload"):
        await _provider(lambda r: httpx.Response(200, json={"nope": 1})).generate(REQUEST)


# --------------------------------------------------------------------------- configuration
def test_settings_load_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CACHE_SIMILARITY_THRESHOLD", "0.85")
    monkeypatch.setenv("CACHE_TTL_SECONDS", "120")
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "http://localhost:3000, https://app.example.com")
    settings = Settings()
    assert settings.cache_similarity_threshold == 0.85
    assert settings.cache_ttl_seconds == 120
    assert settings.cors_allow_origins == ["http://localhost:3000", "https://app.example.com"]


def test_settings_load_from_dotenv_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text("CACHE_SIMILARITY_THRESHOLD=0.95\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert Settings().cache_similarity_threshold == 0.95


def test_secure_defaults() -> None:
    settings = Settings()
    assert settings.llm_provider == "mock"
    assert settings.cors_allow_origins == []
    assert settings.log_prompts is False
    assert settings.host == "127.0.0.1"


@pytest.mark.parametrize("value", ["0", "1.5", "-0.2", "abc"])
def test_invalid_threshold_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("CACHE_SIMILARITY_THRESHOLD", value)
    with pytest.raises(ValidationError):
        Settings()


def test_wildcard_cors_rejected_in_production() -> None:
    with pytest.raises(ValidationError, match="CORS"):
        Settings(app_env="production", cors_allow_origins=["*"])


def test_openai_provider_requires_api_key() -> None:
    with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
        Settings(llm_provider="openai")


def test_secrets_are_masked_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret")
    monkeypatch.setenv("API_KEYS", "client-key-1,client-key-2")
    settings = Settings(llm_provider="openai")
    text = repr(settings) + str(settings.model_dump())
    assert "sk-super-secret" not in text
    assert "client-key-1" not in text
    assert len(settings.api_keys) == 2
