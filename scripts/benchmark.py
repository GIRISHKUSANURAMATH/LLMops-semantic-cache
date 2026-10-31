"""Benchmark: the same request stream WITHOUT cache vs WITH the semantic cache.

    python scripts/benchmark.py
    python scripts/benchmark.py --requests 300 --latency-ms 100 --seed 7

Uses the deterministic mock LLM (fixed artificial latency) and a seeded workload, so runs are
reproducible. Numbers describe this simulated workload only: real hit rates depend entirely
on how repetitive your production traffic is.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import random
import statistics
import time
from pathlib import Path

from llmops_cache.cache.identity import normalize_query
from llmops_cache.cache.service import ChatRequest
from llmops_cache.config import Settings
from llmops_cache.main import build_cache
from llmops_cache.providers import MockLLMProvider

DATASET = Path(__file__).parent / "data" / "semantic_pairs.json"


def build_workload(n: int, seed: int) -> list[str]:
    """Zipf-like stream: a few popular questions asked often, in several phrasings."""
    pairs = json.loads(DATASET.read_text(encoding="utf-8"))["pairs"]
    topics: dict[str, list[str]] = {}
    for p in pairs:
        if p["label"] == "equivalent":
            topics.setdefault(p["a"], [p["a"]]).append(p["b"])
        else:
            topics.setdefault(p["b"], [p["b"]])
    keys = sorted(topics)
    rng = random.Random(seed)  # noqa: S311 - reproducible workload, not security
    weights = [1 / (rank + 1) for rank in range(len(keys))]
    rng.shuffle(keys)
    stream = []
    for _ in range(n):
        topic = rng.choices(keys, weights=weights, k=1)[0]
        stream.append(rng.choice(topics[topic]))
    return stream


async def run_mode(
    stream: list[str], use_cache: bool, latency_ms: int, threshold: float
) -> dict[str, float]:
    settings = Settings(cache_similarity_threshold=threshold, log_level="ERROR")
    llm = MockLLMProvider(latency_ms=latency_ms)
    cache = build_cache(settings, llm=llm)
    latencies = []
    paraphrase_hits = 0
    started = time.perf_counter()
    for message in stream:
        result = await cache.chat(ChatRequest(message=message, model="mock-1", use_cache=use_cache))
        latencies.append(result.latency_ms)
        if result.cache_hit and result.entry_id:
            entry = cache.store.get(result.entry_id)
            if entry and entry.normalized_query != normalize_query(message):
                paraphrase_hits += 1
    wall = time.perf_counter() - started
    snap = cache.metrics.snapshot()
    ordered = sorted(latencies)
    return {
        "requests": len(stream),
        "llm_calls": llm.calls,
        "hit_rate": snap.hit_rate if use_cache else 0.0,
        "paraphrase_hits": paraphrase_hits,
        "mean_latency_ms": statistics.fmean(latencies),
        "p50_latency_ms": ordered[len(ordered) // 2],
        "p95_latency_ms": ordered[int(len(ordered) * 0.95) - 1],
        "wall_time_s": wall,
        "estimated_cost_saved_usd": snap.estimated_cost_saved_usd,
        "estimated_tokens_avoided": snap.estimated_tokens_avoided,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--latency-ms", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stream = build_workload(args.requests, args.seed)
    unique = len({s.lower() for s in stream})
    baseline = asyncio.run(run_mode(stream, False, args.latency_ms, args.threshold))
    cached = asyncio.run(run_mode(stream, True, args.latency_ms, args.threshold))

    print(
        f"Environment: Python {platform.python_version()} on {platform.system()} "
        f"{platform.machine()}, {os.cpu_count()} CPUs"
    )
    print(
        f"Workload: {args.requests} requests, {unique} distinct phrasings, seed={args.seed}; "
        f"mock LLM latency={args.latency_ms} ms; threshold={args.threshold}\n"
    )
    print("| metric | without cache | with semantic cache |\n|---|---:|---:|")
    rows = [
        ("LLM calls", "llm_calls", "{:.0f}"),
        ("cache hit rate", "hit_rate", "{:.1%}"),
        ("  of which: hits on a differently worded query", "paraphrase_hits", "{:.0f}"),
        ("mean latency (ms)", "mean_latency_ms", "{:.2f}"),
        ("p50 latency (ms)", "p50_latency_ms", "{:.2f}"),
        ("p95 latency (ms)", "p95_latency_ms", "{:.2f}"),
        ("total wall time (s)", "wall_time_s", "{:.2f}"),
        ("estimated tokens avoided", "estimated_tokens_avoided", "{:.0f}"),
        ("estimated cost saved (USD)", "estimated_cost_saved_usd", "{:.6f}"),
    ]
    for label, key, fmt in rows:
        print(f"| {label} | {fmt.format(baseline[key])} | {fmt.format(cached[key])} |")
    reduction = 1 - cached["llm_calls"] / baseline["llm_calls"]
    print(
        f"\nLLM calls reduced by {reduction:.1%} (cost estimates use the configured "
        "per-1K-token prices; they are not billing data)."
    )


if __name__ == "__main__":
    main()
