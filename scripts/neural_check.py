"""Measure real similarities with the neural (sentence-transformers) embedder vs the lexical one.

    python -m pip install -e ".[sentence-transformers]"
    python scripts/neural_check.py

The first run downloads the model (~90 MB for all-MiniLM-L6-v2) into the Hugging Face cache.
Exit codes: 0 = measured, 3 = sentence-transformers not installed, 2 = model could not be
loaded (usually: no network for the first download).
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

from llmops_cache.cache.identity import normalize_query
from llmops_cache.embeddings import EmbeddingError, EmbeddingProvider, HashingEmbeddingProvider

PAIRS = [
    ("What is a car?", "Describe an automobile.", "synonyms"),
    ("What is machine learning?", "Explain machine learning.", "rewording"),
    ("Who wrote Hamlet?", "Who is the author of Hamlet?", "paraphrase"),
    (
        "How do I reset my password?",
        "How do I reset my password if I lost access to my email?",
        "related, different answer",
    ),
    ("Convert Celsius to Fahrenheit", "Convert Fahrenheit to Celsius", "related, opposite"),
    ("What is a car?", "What is the capital of France?", "unrelated"),
]


async def similarity(provider: EmbeddingProvider, a: str, b: str) -> float:
    va, vb = await provider.embed_batch([normalize_query(a), normalize_query(b)])
    return float(va @ vb)


async def main() -> int:
    model = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    try:
        from llmops_cache.embeddings.sentence_transformer import (
            SentenceTransformerEmbeddingProvider,
        )

        started = time.perf_counter()
        neural = SentenceTransformerEmbeddingProvider(model, device="cpu")
        load_s = time.perf_counter() - started
    except EmbeddingError as exc:
        message = str(exc)
        print(f"NEURAL MODEL ERROR: {message}")
        if "not installed" in message:
            print("NEURAL MODEL: NOT VERIFIED - sentence-transformers not installed")
            return 3
        cause = exc.__cause__
        print(f"cause: {type(cause).__name__}: {cause}" if cause else "")
        print("NEURAL MODEL: NOT VERIFIED - MODEL DOWNLOAD UNAVAILABLE")
        return 2

    lexical = HashingEmbeddingProvider()
    print(f"model={model} device=cpu dimensions={neural.dimensions} load_time_s={load_s:.1f}")
    print("| pair | kind | neural | lexical (default) |\n|---|---|---:|---:|")
    for a, b, kind in PAIRS:
        n = await similarity(neural, a, b)
        lex = await similarity(lexical, a, b)
        print(f"| {a!r} vs {b!r} | {kind} | {n:.3f} | {lex:.3f} |")
    first = await similarity(neural, *PAIRS[0][:2])
    second = await similarity(neural, *PAIRS[0][:2])
    print(f"deterministic on repeat: {abs(first - second) < 1e-6} ({first:.6f} vs {second:.6f})")
    print("NEURAL MODEL: MEASURED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
