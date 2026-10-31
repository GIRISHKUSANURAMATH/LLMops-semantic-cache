"""Self-contained demo of semantic cache HIT/MISS behaviour.

Runs fully in-process with the mock LLM (no server, no API key, no network):

    python examples/demo.py
    python examples/demo.py --threshold 0.8 --latency-ms 300
"""

from __future__ import annotations

import argparse
import asyncio

from llmops_cache.cache.service import ChatRequest
from llmops_cache.config import Settings
from llmops_cache.main import build_cache
from llmops_cache.providers import MockLLMProvider

QUERIES = [
    ("What is machine learning?", "first time this is asked"),
    ("Explain machine learning to me.", "same meaning, different wording"),
    ("What is machine learning used for?", "shares words, different intent"),
    ("What is the capital of France?", "different semantic meaning"),
    ("How do I reset my password?", "new topic"),
    ("How do I reset my password if I lost access to my email?", "related but NOT the same"),
    ("what is MACHINE LEARNING", "case/punctuation differences"),
]


async def run(threshold: float, latency_ms: int) -> None:
    settings = Settings(
        llm_provider="mock",
        embedding_provider="hashing",
        cache_similarity_threshold=threshold,
        mock_llm_latency_ms=latency_ms,
        log_level="ERROR",
    )
    llm = MockLLMProvider(latency_ms=latency_ms)
    cache = build_cache(settings, llm=llm)

    print(f"Semantic cache demo  (threshold={threshold}, mock LLM latency={latency_ms} ms)")
    print("=" * 78)
    for number, (message, note) in enumerate(QUERIES, start=1):
        request = ChatRequest(message=message, model=settings.llm_default_model)
        nearest = await cache.search(request, top_k=1)  # inspection only, no LLM call
        calls_before = llm.calls
        result = await cache.chat(request)
        llm_called = llm.calls > calls_before
        status = "CACHE HIT " if result.cache_hit else "CACHE MISS"
        best = nearest[0][0].similarity if nearest else None
        similarity = f"{best:.3f}" if best is not None else "  -  "
        print(f"Request {number}: {message!r}")
        print(
            f"   {status} | LLM {'called' if llm_called else 'NOT called':<10} | "
            f"nearest sim {similarity} | {result.latency_ms:8.2f} ms | ({note})"
        )
    print("=" * 78)
    print("'nearest sim' = cosine similarity to the closest cached entry before the request;")
    print(f"a request is a HIT only when it is >= the threshold ({threshold}).")
    print()

    snap = cache.metrics.snapshot()
    print(
        f"Requests: {snap.requests_total}   hits: {snap.cache_hits}   "
        f"misses: {snap.cache_misses}   hit rate: {snap.hit_rate:.0%}"
    )
    print(f"LLM calls made: {snap.llm_calls}   avoided: {snap.llm_calls_avoided}")
    print(
        f"Avg miss latency: {snap.average_miss_latency_ms} ms   "
        f"avg hit latency: {snap.average_hit_latency_ms} ms"
    )
    print(
        f"Estimated tokens avoided: {snap.estimated_tokens_avoided}   "
        f"estimated cost saved: ${snap.estimated_cost_saved_usd:.6f} (estimate)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--latency-ms", type=int, default=600)
    args = parser.parse_args()
    asyncio.run(run(args.threshold, args.latency_ms))


if __name__ == "__main__":
    main()
