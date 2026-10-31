"""Benchmark one request stream three ways: no cache, exact-match cache, semantic cache.

    python scripts/benchmark.py
    python scripts/benchmark.py --requests 300 --latency-ms 100 --seed 7
    python scripts/benchmark.py --embedding-provider sentence-transformers

Every request carries a hidden *topic* label (which answer it actually needs). That lets the
benchmark split cache hits into:

* exact-repeat hits   - same normalised text was asked before (an exact-match cache gets these)
* reworded hits       - different wording, SAME topic: the genuine semantic-cache benefit
* false-positive hits - different wording, DIFFERENT topic: a wrong answer was served

and misses into:

* first-time misses   - the topic had never been seen; nothing could have helped
* missed rewordings   - the topic was cached under other wording, but similarity < threshold

Uses the deterministic mock LLM and a seeded workload, so runs are reproducible. Results
describe this synthetic workload only; real traffic has its own repeat rate.
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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from llmops_cache.cache.identity import normalize_query
from llmops_cache.cache.service import ChatRequest
from llmops_cache.config import Settings
from llmops_cache.main import build_cache
from llmops_cache.providers import MockLLMProvider

DATASET = Path(__file__).parent / "data" / "semantic_pairs.json"
SWEEP = (0.80, 0.85, 0.90, 0.95)


def load_topics() -> dict[str, list[str]]:
    """Topic -> phrasings. Equivalent pairs share a topic; every other phrasing is its own topic.

    A phrasing belongs to exactly one topic (deduplicated after normalisation), so a request is
    never "its own false positive".
    """
    pairs = json.loads(DATASET.read_text(encoding="utf-8"))["pairs"]
    topics: dict[str, list[str]] = {}
    owner: dict[str, str] = {}

    def add(topic: str, phrasing: str) -> None:
        key = normalize_query(phrasing)
        if key in owner:
            return
        owner[key] = topic
        topics.setdefault(topic, []).append(phrasing)

    for p in pairs:
        if p["label"] == "equivalent":
            topic = owner.get(normalize_query(p["a"]), p["a"])
            add(topic, p["a"])
            add(topic, p["b"])
    for p in pairs:
        if p["label"] != "equivalent":
            for phrasing in (p["a"], p["b"]):
                add(phrasing, phrasing)
    return topics


def build_workload(n: int, seed: int) -> list[tuple[str, str]]:
    """Zipf-like stream of (topic, phrasing): a few topics asked often, in several wordings."""
    topics = load_topics()
    keys = sorted(topics)
    rng = random.Random(seed)  # noqa: S311 - reproducible workload, not security
    rng.shuffle(keys)
    weights = [1 / (rank + 1) for rank in range(len(keys))]
    stream = []
    for _ in range(n):
        topic = rng.choices(keys, weights=weights, k=1)[0]
        stream.append((topic, rng.choice(topics[topic])))
    return stream


def describe_workload(stream: list[tuple[str, str]]) -> dict[str, int]:
    seen_text: set[str] = set()
    seen_topic: set[str] = set()
    counts = {"requests": len(stream), "exact_repeats": 0, "reworded": 0, "new_topic": 0}
    for topic, message in stream:
        text = normalize_query(message)
        if text in seen_text:
            counts["exact_repeats"] += 1
        elif topic in seen_topic:
            counts["reworded"] += 1
        else:
            counts["new_topic"] += 1
        seen_text.add(text)
        seen_topic.add(topic)
    counts["distinct_phrasings"] = len(seen_text)
    return counts


@dataclass
class Outcome:
    llm_calls: int = 0
    exact_hits: int = 0
    reworded_hits: int = 0
    false_positive_hits: int = 0
    first_time_misses: int = 0
    missed_rewordings: int = 0
    latencies: list[float] = field(default_factory=list)
    wall_time_s: float = 0.0
    tokens_avoided: int = 0
    cost_saved: float = 0.0
    false_positive_examples: list[tuple[str, str, float]] = field(default_factory=list)

    @property
    def hits(self) -> int:
        return self.exact_hits + self.reworded_hits + self.false_positive_hits


async def run(
    stream: list[tuple[str, str]],
    *,
    use_cache: bool,
    latency_ms: int,
    threshold: float,
    embedding_provider: str,
) -> Outcome:
    settings = Settings(
        cache_similarity_threshold=threshold,
        embedding_provider=embedding_provider,  # type: ignore[arg-type]
        log_level="ERROR",
    )
    llm = MockLLMProvider(latency_ms=latency_ms)
    cache = build_cache(settings, llm=llm)
    topic_of_entry: dict[str, str] = {}
    seen_topic: set[str] = set()
    seen_text: set[str] = set()
    out = Outcome()
    started = time.perf_counter()
    for topic, message in stream:
        text = normalize_query(message)
        result = await cache.chat(ChatRequest(message=message, model="mock-1", use_cache=use_cache))
        out.latencies.append(result.latency_ms)
        if result.cache_hit and result.entry_id:
            entry = cache.store.get(result.entry_id)
            if topic_of_entry.get(result.entry_id) != topic:
                out.false_positive_hits += 1
                cached = entry.query if entry else "?"
                out.false_positive_examples.append(
                    (message, cached, float(result.similarity or 0.0))
                )
            elif text in seen_text:
                out.exact_hits += 1
            else:
                out.reworded_hits += 1
        elif not result.cache_hit:
            if result.entry_id:
                topic_of_entry[result.entry_id] = topic
            if use_cache and topic in seen_topic:
                out.missed_rewordings += 1
            else:
                out.first_time_misses += 1
        seen_topic.add(topic)
        seen_text.add(text)
    out.wall_time_s = time.perf_counter() - started
    out.llm_calls = llm.calls
    snap = cache.metrics.snapshot()
    out.tokens_avoided = snap.estimated_tokens_avoided
    out.cost_saved = snap.estimated_cost_saved_usd
    await cache.embedder.aclose()
    return out


def _p50(values: list[float]) -> float:
    return sorted(values)[len(values) // 2]


def _p95(values: list[float]) -> float:
    return sorted(values)[max(0, int(len(values) * 0.95) - 1)]


def _pct(part: int, whole: int) -> str:
    return f"{part / whole:.1%}" if whole else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--latency-ms", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--embedding-provider",
        default="hashing",
        choices=["hashing", "sentence-transformers", "openai"],
    )
    args = parser.parse_args()

    stream = build_workload(args.requests, args.seed)
    w = describe_workload(stream)
    common = {
        "latency_ms": args.latency_ms,
        "threshold": args.threshold,
        "embedding_provider": args.embedding_provider,
    }
    none = asyncio.run(run(stream, use_cache=False, **common))
    semantic = asyncio.run(run(stream, use_cache=True, **common))
    n = w["requests"]

    print(
        f"Environment: Python {platform.python_version()} on {platform.system()} "
        f"{platform.release()} {platform.machine()}, {os.cpu_count()} CPUs"
    )
    print(
        f"Embedding provider: {args.embedding_provider}; threshold={args.threshold}; "
        f"mock LLM latency={args.latency_ms} ms; seed={args.seed}\n"
    )

    print("## Workload\n")
    print(
        "| requests | distinct phrasings | exact repeats | reworded (topic seen, new wording) "
        "| new topic |\n|---:|---:|---:|---:|---:|"
    )
    print(
        f"| {n} | {w['distinct_phrasings']} | {w['exact_repeats']} ({_pct(w['exact_repeats'], n)})"
        f" | {w['reworded']} ({_pct(w['reworded'], n)}) | {w['new_topic']} |\n"
    )

    print("## LLM calls\n")
    print("| strategy | LLM calls | calls avoided vs no cache |\n|---|---:|---:|")
    print(f"| no cache | {none.llm_calls} | 0 |")
    exact_calls = w["distinct_phrasings"]
    print(
        f"| exact-match cache (computed: one call per distinct phrasing) | {exact_calls} "
        f"| {n - exact_calls} |"
    )
    print(f"| semantic cache (measured) | {semantic.llm_calls} | {n - semantic.llm_calls} |\n")

    print("## Semantic cache outcomes\n")
    print("| outcome | count | share of requests |\n|---|---:|---:|")
    rows = [
        (
            "exact-repeat hits (text asked before; an exact-match cache gets these too)",
            semantic.exact_hits,
        ),
        (
            "reworded hits: new wording, correct answer (the semantic benefit)",
            semantic.reworded_hits,
        ),
        ("false-positive hits (WRONG answer served)", semantic.false_positive_hits),
        ("first-time misses (unavoidable)", semantic.first_time_misses),
        ("missed rewordings (same topic, similarity below threshold)", semantic.missed_rewordings),
    ]
    for label, value in rows:
        print(f"| {label} | {value} | {_pct(value, n)} |")
    print(
        f"| **overall cache hit rate (includes exact repeats)** | {semantic.hits} "
        f"| {_pct(semantic.hits, n)} |"
    )
    rewording_total = semantic.reworded_hits + semantic.missed_rewordings
    print(
        f"\nReworded requests served correctly: {semantic.reworded_hits}/{rewording_total} "
        f"({_pct(semantic.reworded_hits, rewording_total)})"
    )
    for asked, cached, sim in semantic.false_positive_examples[:5]:
        print(
            f"  false positive: asked {asked!r} -> served answer cached for {cached!r} "
            f"(similarity {sim:.3f})"
        )

    print("\n## Latency\n")
    print("| metric | no cache | semantic cache |\n|---|---:|---:|")
    stats: list[tuple[str, Callable[[list[float]], float]]] = [
        ("mean (ms)", statistics.fmean),
        ("p50 (ms)", _p50),
        ("p95 (ms)", _p95),
    ]
    for label, fn in stats:
        print(f"| {label} | {fn(none.latencies):.2f} | {fn(semantic.latencies):.2f} |")
    print(f"| total wall time (s) | {none.wall_time_s:.2f} | {semantic.wall_time_s:.2f} |")
    print(
        f"\nEstimated tokens avoided: {semantic.tokens_avoided}; estimated cost saved: "
        f"${semantic.cost_saved:.6f} (configured prices x estimated tokens; not billing data)"
    )

    print("\n## Threshold sweep (same workload, LLM latency 0)\n")
    print(
        "| threshold | LLM calls | exact hits | reworded hits | false positives "
        "| missed rewordings |\n|---:|---:|---:|---:|---:|---:|"
    )
    for threshold in SWEEP:
        o = asyncio.run(
            run(
                stream,
                use_cache=True,
                latency_ms=0,
                threshold=threshold,
                embedding_provider=args.embedding_provider,
            )
        )
        print(
            f"| {threshold:.2f} | {o.llm_calls} | {o.exact_hits} | {o.reworded_hits} | "
            f"{o.false_positive_hits} | {o.missed_rewordings} |"
        )


if __name__ == "__main__":
    main()
