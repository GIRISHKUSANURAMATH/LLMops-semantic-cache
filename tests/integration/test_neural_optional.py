"""Real sentence-transformers model. Opt-in: needs the extra installed and a one-time download.

RUN_NEURAL_TESTS=1 pytest -m neural          (macOS / Linux)
$env:RUN_NEURAL_TESTS=1; pytest -m neural    (Windows PowerShell)
"""

from __future__ import annotations

import os

import pytest

from llmops_cache.cache.identity import normalize_query
from llmops_cache.embeddings import HashingEmbeddingProvider
from llmops_cache.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider

pytestmark = [
    pytest.mark.neural,
    pytest.mark.skipif(
        os.environ.get("RUN_NEURAL_TESTS") != "1",
        reason="set RUN_NEURAL_TESTS=1 to run tests that download a real embedding model",
    ),
]

MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


async def _sim(provider, a: str, b: str) -> float:  # type: ignore[no-untyped-def]
    va, vb = await provider.embed_batch([normalize_query(a), normalize_query(b)])
    return float(va @ vb)


async def test_neural_model_understands_synonyms_better_than_lexical() -> None:
    neural = SentenceTransformerEmbeddingProvider(MODEL)
    lexical = HashingEmbeddingProvider()
    car_neural = await _sim(neural, "What is a car?", "Describe an automobile.")
    car_lexical = await _sim(lexical, "What is a car?", "Describe an automobile.")
    unrelated = await _sim(neural, "What is a car?", "What is the capital of France?")
    print(
        f"car/automobile neural={car_neural:.3f} lexical={car_lexical:.3f} "
        f"unrelated={unrelated:.3f}"
    )
    assert car_neural > car_lexical
    assert car_neural > unrelated


async def test_neural_model_is_deterministic_on_cpu() -> None:
    provider = SentenceTransformerEmbeddingProvider(MODEL)
    first = await _sim(provider, "What is machine learning?", "Explain machine learning.")
    second = await _sim(provider, "What is machine learning?", "Explain machine learning.")
    assert first == pytest.approx(second, abs=1e-6)
    assert 0.0 < first <= 1.0
