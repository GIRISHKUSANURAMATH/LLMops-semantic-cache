"""In-process metrics with a JSON view and a Prometheus exposition.

Cost figures are *estimates*: tokens a cached answer would have consumed, priced with the
configured per-1K-token rates. They are not provider billing data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

Outcome = Literal["hit", "miss", "coalesced", "bypass", "error"]

_LATENCY_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)


@dataclass
class _LatencyStat:
    count: int = 0
    total_ms: float = 0.0

    def add(self, value_ms: float) -> None:
        self.count += 1
        self.total_ms += value_ms

    @property
    def average_ms(self) -> float | None:
        return round(self.total_ms / self.count, 3) if self.count else None


@dataclass
class MetricsSnapshot:
    requests_total: int
    cache_hits: int
    cache_misses: int
    coalesced_requests: int
    cache_bypassed: int
    errors: int
    hit_rate: float
    llm_calls: int
    llm_calls_avoided: int
    estimated_tokens_avoided: int
    estimated_cost_saved_usd: float
    average_latency_ms: float | None
    average_hit_latency_ms: float | None
    average_miss_latency_ms: float | None
    entries_count: int


@dataclass
class Metrics:
    cost_per_1k_input: float
    cost_per_1k_output: float
    registry: CollectorRegistry = field(default_factory=CollectorRegistry)

    def __post_init__(self) -> None:
        self._counts: dict[str, int] = dict.fromkeys(
            ("hit", "miss", "coalesced", "bypass", "error"), 0
        )
        self._latency: dict[str, _LatencyStat] = {}
        self._all_latency = _LatencyStat()
        self.llm_calls = 0
        self.tokens_avoided = 0
        self.cost_saved = 0.0
        self.entries_count = 0

        self._p_requests = Counter(
            "llmops_requests", "Chat requests by outcome", ["outcome"], registry=self.registry
        )
        self._p_latency = Histogram(
            "llmops_request_latency_seconds",
            "End-to-end chat latency by outcome",
            ["outcome"],
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self._p_llm_calls = Counter(
            "llmops_llm_calls", "Upstream LLM calls made", registry=self.registry
        )
        self._p_tokens_avoided = Counter(
            "llmops_estimated_tokens_avoided",
            "Estimated tokens not sent upstream",
            registry=self.registry,
        )
        self._p_cost_saved = Counter(
            "llmops_estimated_cost_saved_usd",
            "Estimated USD saved (not billing data)",
            registry=self.registry,
        )
        self._p_entries = Gauge(
            "llmops_cache_entries", "Entries currently in the cache", registry=self.registry
        )

    def record(self, outcome: Outcome, latency_ms: float) -> None:
        self._counts[outcome] += 1
        self._latency.setdefault(outcome, _LatencyStat()).add(latency_ms)
        self._all_latency.add(latency_ms)
        self._p_requests.labels(outcome=outcome).inc()
        self._p_latency.labels(outcome=outcome).observe(latency_ms / 1000)

    def record_llm_call(self) -> None:
        self.llm_calls += 1
        self._p_llm_calls.inc()

    def record_avoided(self, input_tokens: int, output_tokens: int) -> float:
        saved = (input_tokens / 1000) * self.cost_per_1k_input + (
            output_tokens / 1000
        ) * self.cost_per_1k_output
        self.tokens_avoided += input_tokens + output_tokens
        self.cost_saved += saved
        self._p_tokens_avoided.inc(input_tokens + output_tokens)
        self._p_cost_saved.inc(saved)
        return saved

    def set_entries(self, count: int) -> None:
        self.entries_count = count
        self._p_entries.set(count)

    def reset(self) -> None:
        """Forget counters (used by tests/benchmarks). Prometheus counters stay monotonic."""
        self._counts = dict.fromkeys(self._counts, 0)
        self._latency.clear()
        self._all_latency = _LatencyStat()
        self.llm_calls = 0
        self.tokens_avoided = 0
        self.cost_saved = 0.0

    def snapshot(self) -> MetricsSnapshot:
        hits = self._counts["hit"] + self._counts["coalesced"]
        lookups = hits + self._counts["miss"]
        total = sum(self._counts.values())

        def avg(outcome: str) -> float | None:
            stat = self._latency.get(outcome)
            return stat.average_ms if stat else None

        return MetricsSnapshot(
            requests_total=total,
            cache_hits=self._counts["hit"],
            cache_misses=self._counts["miss"],
            coalesced_requests=self._counts["coalesced"],
            cache_bypassed=self._counts["bypass"],
            errors=self._counts["error"],
            hit_rate=round(hits / lookups, 4) if lookups else 0.0,
            llm_calls=self.llm_calls,
            llm_calls_avoided=hits,
            estimated_tokens_avoided=self.tokens_avoided,
            estimated_cost_saved_usd=round(self.cost_saved, 6),
            average_latency_ms=self._all_latency.average_ms,
            average_hit_latency_ms=avg("hit"),
            average_miss_latency_ms=avg("miss"),
            entries_count=self.entries_count,
        )
