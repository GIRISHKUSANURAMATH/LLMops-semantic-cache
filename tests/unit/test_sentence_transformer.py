"""The sentence-transformers provider, tested with a fake model (no download, no torch).

A real-model check lives in tests/integration/test_neural_optional.py and only runs when
RUN_NEURAL_TESTS=1.
"""

from __future__ import annotations

import sys
import types
from typing import ClassVar

import numpy as np
import pytest

from llmops_cache.cache.service import CacheConfig, ChatRequest, SemanticCache
from llmops_cache.cache.vector_store import InMemoryVectorStore
from llmops_cache.config import Settings
from llmops_cache.embeddings import EmbeddingError, build_embedding_provider
from llmops_cache.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider
from llmops_cache.observability.metrics import Metrics
from llmops_cache.providers import MockLLMProvider

from helpers import FakeClock

# Deliberately NOT unit length, to prove the provider normalises.
_VECTORS = {
    "what is a car": [3.0, 4.0, 0.0],
    "describe an automobile": [3.0, 3.9, 0.5],
    "capital of france": [0.0, 0.0, 7.0],
}


class _FakeModel:
    instances: ClassVar[list[_FakeModel]] = []
    declared_dims: int | None = 3
    fail_load: Exception | None = None
    fail_encode: Exception | None = None

    def __init__(self, model_name: str, device: str | None = None) -> None:
        if _FakeModel.fail_load is not None:
            raise _FakeModel.fail_load
        self.model_name = model_name
        self.device = device
        self.encode_calls: list[dict[str, object]] = []
        _FakeModel.instances.append(self)

    def get_sentence_embedding_dimension(self) -> int | None:
        return _FakeModel.declared_dims

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        if _FakeModel.fail_encode is not None:
            raise _FakeModel.fail_encode
        self.encode_calls.append(kwargs)
        return np.array([_VECTORS.get(t, [1.0, 0.0, 0.0]) for t in texts], dtype=np.float64)


@pytest.fixture
def fake_st(monkeypatch: pytest.MonkeyPatch) -> type[_FakeModel]:
    _FakeModel.instances = []
    _FakeModel.declared_dims = 3
    _FakeModel.fail_load = None
    _FakeModel.fail_encode = None
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return _FakeModel


async def test_loads_model_on_cpu_and_reports_dimensions(fake_st) -> None:
    provider = SentenceTransformerEmbeddingProvider("my-model")
    assert provider.dimensions == 3
    assert provider.name == "st-my-model"
    assert fake_st.instances[0].device == "cpu"
    assert fake_st.instances[0].model_name == "my-model"


async def test_vectors_are_float32_unit_length(fake_st) -> None:
    provider = SentenceTransformerEmbeddingProvider("m")
    vectors = await provider.embed_batch(["what is a car", "capital of france"])
    for vector in vectors:
        assert vector.dtype == np.float32
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(vectors[0], [0.6, 0.8, 0.0], rtol=1e-6)
    assert fake_st.instances[0].encode_calls[0]["normalize_embeddings"] is True
    assert fake_st.instances[0].encode_calls[0]["show_progress_bar"] is False


async def test_similarity_is_cosine(fake_st) -> None:
    provider = SentenceTransformerEmbeddingProvider("m")
    a, b, c = await provider.embed_batch(
        ["what is a car", "describe an automobile", "capital of france"]
    )
    expected = np.dot([3, 4, 0], [3, 3.9, 0.5]) / (5 * np.linalg.norm([3, 3.9, 0.5]))
    assert float(a @ b) == pytest.approx(expected, abs=1e-6)
    assert float(a @ c) == pytest.approx(0.0, abs=1e-6)


async def test_measures_dimensions_when_model_does_not_declare_them(fake_st) -> None:
    fake_st.declared_dims = None
    assert SentenceTransformerEmbeddingProvider("m").dimensions == 3


async def test_device_is_configurable_via_settings(fake_st) -> None:
    settings = Settings(embedding_provider="sentence-transformers", embedding_device="cuda")
    provider = build_embedding_provider(settings)
    assert isinstance(provider, SentenceTransformerEmbeddingProvider)
    assert fake_st.instances[0].device == "cuda"
    assert fake_st.instances[0].model_name == settings.embedding_model


def test_missing_package_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # makes import fail
    with pytest.raises(EmbeddingError, match=r'pip install -e "\.\[sentence-transformers\]"'):
        SentenceTransformerEmbeddingProvider("m")


def test_model_download_failure_is_a_clear_error(fake_st) -> None:
    fake_st.fail_load = OSError("We couldn't connect to 'https://huggingface.co'")
    with pytest.raises(EmbeddingError, match="network access once"):
        SentenceTransformerEmbeddingProvider("m")


async def test_encode_failure_raises_embedding_error(fake_st) -> None:
    provider = SentenceTransformerEmbeddingProvider("m")
    fake_st.fail_encode = RuntimeError("out of memory")
    with pytest.raises(EmbeddingError):
        await provider.embed_text("x")


async def test_empty_batch_does_not_call_model(fake_st) -> None:
    provider = SentenceTransformerEmbeddingProvider("m")
    assert await provider.embed_batch([]) == []
    assert fake_st.instances[0].encode_calls == []


async def test_semantic_cache_works_end_to_end_with_neural_provider(fake_st) -> None:
    llm = MockLLMProvider(latency_ms=0)
    cache = SemanticCache(
        embedder=SentenceTransformerEmbeddingProvider("m"),
        store=InMemoryVectorStore(),
        llm=llm,
        metrics=Metrics(cost_per_1k_input=0, cost_per_1k_output=0),
        config=CacheConfig(similarity_threshold=0.9),
        clock=FakeClock(),
    )
    await cache.chat(ChatRequest(message="What is a car?", model="m"))
    hit = await cache.chat(ChatRequest(message="Describe an automobile", model="m"))
    miss = await cache.chat(ChatRequest(message="Capital of France", model="m"))
    assert hit.cache_hit is True  # a synonym pair the lexical embedder scores 0.0 on
    assert miss.cache_hit is False
    assert llm.calls == 2
