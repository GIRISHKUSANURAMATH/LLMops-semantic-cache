from __future__ import annotations

import httpx
import numpy as np
import pytest

from llmops_cache.cache.identity import context_key, fingerprint, normalize_query
from llmops_cache.embeddings import EmbeddingError, HashingEmbeddingProvider
from llmops_cache.embeddings.openai import OpenAIEmbeddingProvider


# --------------------------------------------------------------------------- normalisation
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  What is   Machine Learning?? ", "what is machine learning"),
        ("\uff21\uff22\uff23", "abc"),  # NFKC folds full-width characters
        ("line\nbreak\ttab", "line break tab"),
        ("Keep inner punctuation, please.", "keep inner punctuation, please"),
    ],
)
def test_normalize_query(raw: str, expected: str) -> None:
    assert normalize_query(raw) == expected


def _key(**overrides: object) -> str:
    params: dict[str, object] = {
        "provider": "mock",
        "model": "m",
        "system_prompt": None,
        "temperature": 0.0,
        "max_tokens": 512,
        "namespace": None,
    }
    params.update(overrides)
    return context_key(**params)  # type: ignore[arg-type]


def test_context_key_is_stable_and_sensitive_to_every_field() -> None:
    base = _key()
    assert base == _key()
    for field, value in [
        ("provider", "openai"),
        ("model", "m2"),
        ("system_prompt", "x"),
        ("temperature", 0.5),
        ("max_tokens", 10),
        ("namespace", "t1"),
    ]:
        assert _key(**{field: value}) != base, field


def test_fingerprint_does_not_reveal_text() -> None:
    fp = fingerprint("my password is hunter2")
    assert "hunter2" not in fp
    assert len(fp) == 12


# --------------------------------------------------------------------------- hashing embedder
async def _sim(embedder: HashingEmbeddingProvider, a: str, b: str) -> float:
    va, vb = await embedder.embed_batch([a, b])
    return float(va @ vb)


async def test_hashing_embeddings_are_unit_length_and_deterministic() -> None:
    embedder = HashingEmbeddingProvider(dimensions=256)
    first = await embedder.embed_text("what is machine learning")
    second = await HashingEmbeddingProvider(dimensions=256).embed_text("what is machine learning")
    assert first.shape == (256,)
    assert first.dtype == np.float32
    assert np.linalg.norm(first) == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_array_equal(first, second)


async def test_batch_matches_single() -> None:
    embedder = HashingEmbeddingProvider()
    batch = await embedder.embed_batch(["alpha beta", "gamma"])
    np.testing.assert_array_equal(batch[1], await embedder.embed_text("gamma"))


async def test_paraphrase_scores_higher_than_unrelated() -> None:
    embedder = HashingEmbeddingProvider()
    paraphrase = await _sim(embedder, "what is machine learning", "explain machine learning to me")
    unrelated = await _sim(embedder, "what is machine learning", "capital of france")
    assert paraphrase >= 0.9
    assert unrelated < 0.2


async def test_stopword_only_query_still_gets_a_vector() -> None:
    vector = await HashingEmbeddingProvider().embed_text("what is it")
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


# --------------------------------------------------------------------------- openai embeddings
def _openai_embedder(handler) -> OpenAIEmbeddingProvider:  # type: ignore[no-untyped-def]
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://x/v1")
    return OpenAIEmbeddingProvider(
        api_key="k", model="text-embedding-3-small", base_url="https://x/v1", client=client
    )


async def test_openai_embeddings_parse_and_normalise() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 2.0]},
                    {"index": 0, "embedding": [3.0, 4.0]},
                ]
            },
        )

    vectors = await _openai_embedder(handler).embed_batch(["a", "b"])
    np.testing.assert_allclose(vectors[0], [0.6, 0.8], rtol=1e-6)
    np.testing.assert_allclose(vectors[1], [0.0, 1.0], rtol=1e-6)


async def test_openai_embedding_failure_raises_embedding_error() -> None:
    embedder = _openai_embedder(lambda request: httpx.Response(500, json={}))
    with pytest.raises(EmbeddingError):
        await embedder.embed_text("a")
