"""Evaluate how similarity thresholds behave on a labelled query-pair dataset.

    python scripts/evaluate_threshold.py
    python scripts/evaluate_threshold.py --embedding-provider sentence-transformers

For each pair the second query is treated as an incoming request and the first as a cached
entry. A pair is a *correct hit* if it is labelled ``equivalent`` and similarity >= threshold;
a *false positive* (wrong answer served from cache) if it is ``related``/``unrelated`` and
similarity >= threshold.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from llmops_cache.cache.identity import normalize_query
from llmops_cache.config import Settings
from llmops_cache.embeddings import build_embedding_provider

DATASET = Path(__file__).parent / "data" / "semantic_pairs.json"
THRESHOLDS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


async def score_pairs(provider_name: str) -> tuple[str, list[dict[str, object]]]:
    settings = Settings(embedding_provider=provider_name)  # type: ignore[arg-type]
    embedder = build_embedding_provider(settings)
    pairs = json.loads(DATASET.read_text(encoding="utf-8"))["pairs"]
    scored = []
    for pair in pairs:
        va, vb = await embedder.embed_batch(
            [normalize_query(pair["a"]), normalize_query(pair["b"])]
        )
        scored.append({**pair, "similarity": float(va @ vb)})
    await embedder.aclose()
    return embedder.name, scored


def summarise(scored: list[dict[str, object]], threshold: float) -> dict[str, float]:
    def hit(p: dict[str, object]) -> bool:
        return float(p["similarity"]) >= threshold  # type: ignore[arg-type]

    equivalent = [p for p in scored if p["label"] == "equivalent"]
    negatives = [p for p in scored if p["label"] != "equivalent"]
    true_hits = sum(hit(p) for p in equivalent)
    false_hits = sum(hit(p) for p in negatives)
    served = true_hits + false_hits
    return {
        "threshold": threshold,
        "recall": true_hits / len(equivalent),
        "precision": true_hits / served if served else 1.0,
        "false_positives": false_hits,
        "related_false_positives": sum(hit(p) for p in negatives if p["label"] == "related"),
        "unrelated_false_positives": sum(hit(p) for p in negatives if p["label"] == "unrelated"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--embedding-provider",
        default="hashing",
        choices=["hashing", "sentence-transformers", "openai"],
    )
    parser.add_argument("--show-pairs", action="store_true")
    args = parser.parse_args()

    name, scored = asyncio.run(score_pairs(args.embedding_provider))
    counts = {
        label: sum(p["label"] == label for p in scored)
        for label in ("equivalent", "related", "unrelated")
    }
    print(f"Embedding provider: {name}   pairs: {len(scored)} {counts}\n")

    if args.show_pairs:
        print("| label | similarity | query A | query B |\n|---|---:|---|---|")
        for p in sorted(scored, key=lambda item: -float(item["similarity"])):  # type: ignore[arg-type]
            print(f"| {p['label']} | {float(p['similarity']):.3f} | {p['a']} | {p['b']} |")  # type: ignore[arg-type]
        print()

    print(
        "| threshold | recall (equivalent served) | precision | false positives "
        "(related / unrelated) |"
    )
    print("|---:|---:|---:|---:|")
    for threshold in THRESHOLDS:
        s = summarise(scored, threshold)
        print(
            f"| {threshold:.2f} | {s['recall']:.0%} | {s['precision']:.0%} | "
            f"{s['false_positives']:.0f} ({s['related_false_positives']:.0f} / "
            f"{s['unrelated_false_positives']:.0f}) |"
        )


if __name__ == "__main__":
    main()
