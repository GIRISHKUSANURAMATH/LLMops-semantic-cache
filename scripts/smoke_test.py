"""Black-box smoke test against a RUNNING server, over real HTTP.

    python scripts/smoke_test.py --url http://127.0.0.1:8000
    python scripts/smoke_test.py --url http://127.0.0.1:8000 --phase persist-seed
    (restart the server with the same VECTOR_STORE_PATH)
    python scripts/smoke_test.py --url http://127.0.0.1:8000 --phase persist-check

Every check prints PASS or FAIL with the evidence (status codes, response bodies, LLM-call
deltas read from /v1/metrics). Exit code 0 only if every check passed.
Requires the server to use the mock LLM or a real one; it never needs an API key itself unless
the server has API_KEYS set (then pass --api-key).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import threading
import time
from typing import Any

import httpx

MISS_Q = "What is machine learning?"
HIT_Q = "Explain machine learning to me."
DIFF_Q = "What is the capital of France?"


class Smoke:
    def __init__(self, url: str, api_key: str | None) -> None:
        self.url = url.rstrip("/")
        self.headers = {"X-API-Key": api_key} if api_key else {}
        self.client = httpx.Client(base_url=self.url, headers=self.headers, timeout=30)
        self.failures = 0

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        print(f"{'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
        if not ok:
            self.failures += 1
        return ok

    def llm_calls(self) -> int:
        return int(self.client.get("/v1/metrics").json()["llm_calls"])

    def chat(self, message: str, **extra: Any) -> tuple[dict[str, Any], int]:
        """POST /v1/chat; returns (body, LLM calls this request caused)."""
        before = self.llm_calls()
        response = self.client.post("/v1/chat", json={"message": message, **extra})
        body = response.json()
        print(f"      POST /v1/chat {message!r} -> HTTP {response.status_code} {json.dumps(body)}")
        return body, self.llm_calls() - before

    # ------------------------------------------------------------------ phases
    def endpoints(self) -> None:
        for path in (
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/v1/cache/stats",
            "/v1/metrics",
            "/metrics",
        ):
            response = self.client.get(path)
            self.check(f"GET {path}", response.status_code == 200, f"HTTP {response.status_code}")
        print(f"      /health -> {self.client.get('/health').text}")

    def miss_hit_different(self) -> None:
        cleared = self.client.delete("/v1/cache")
        self.check("DELETE /v1/cache", cleared.status_code == 200, cleared.text)
        body, calls = self.chat(MISS_Q)
        self.check(
            "request 1 is a CACHE MISS and calls the LLM",
            body.get("cache_hit") is False and calls == 1,
            f"cache_hit={body.get('cache_hit')} llm_calls_delta={calls}",
        )
        body, calls = self.chat(HIT_Q)
        self.check(
            "request 2 (reworded) is a CACHE HIT and does NOT call the LLM",
            body.get("cache_hit") is True and calls == 0,
            f"cache_hit={body.get('cache_hit')} similarity={body.get('similarity')} "
            f"llm_calls_delta={calls}",
        )
        body, calls = self.chat(DIFF_Q)
        self.check(
            "request 3 (different question) is a CACHE MISS and calls the LLM",
            body.get("cache_hit") is False and calls == 1,
            f"cache_hit={body.get('cache_hit')} llm_calls_delta={calls}",
        )
        stats = self.client.get("/v1/cache/stats").json()
        print(f"      /v1/cache/stats -> {json.dumps(stats)}")
        self.check(
            "stats: 2 entries",
            stats.get("entries_count") == 2,
            f"entries_count={stats.get('entries_count')}",
        )
        metrics = self.client.get("/v1/metrics").json()
        print(f"      /v1/metrics -> {json.dumps(metrics)}")
        self.check(
            "metrics: at least 1 hit and 2 misses",
            metrics.get("cache_hits", 0) >= 1 and metrics.get("cache_misses", 0) >= 2,
        )

    def ttl(self, ttl_seconds: int) -> None:
        message = "TTL probe: how do volcanoes form?"
        body, calls = self.chat(message, ttl_seconds=ttl_seconds)
        self.check("TTL: insert entry (miss)", body.get("cache_hit") is False and calls == 1)
        body, calls = self.chat(message)
        self.check(
            "TTL: before expiry -> HIT",
            body.get("cache_hit") is True and calls == 0,
            f"cache_hit={body.get('cache_hit')}",
        )
        wait = ttl_seconds + 1.5
        print(f"      waiting {wait:.1f}s for the {ttl_seconds}s TTL to expire...")
        time.sleep(wait)
        body, calls = self.chat(message)
        self.check(
            "TTL: after expiry -> MISS and LLM called again",
            body.get("cache_hit") is False and calls == 1,
            f"cache_hit={body.get('cache_hit')} llm_calls_delta={calls}",
        )

    def concurrency(self, n: int) -> None:
        message = f"Concurrency probe {time.time_ns()}: what is a hash table?"
        before = self.llm_calls()
        barrier = threading.Barrier(n)

        def send(_: int) -> dict[str, Any]:
            with httpx.Client(base_url=self.url, headers=self.headers, timeout=30) as client:
                barrier.wait()
                return dict(client.post("/v1/chat", json={"message": message}).json())

        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            results = list(pool.map(send, range(n)))
        calls = self.llm_calls() - before
        sources = [r.get("source") for r in results]
        print(f"      {n} simultaneous identical requests -> sources={sources}")
        self.check(
            f"concurrency: {n} identical simultaneous requests -> exactly 1 LLM call",
            calls == 1,
            f"llm_calls_delta={calls}",
        )
        self.check(
            "concurrency: all responses identical", len({r.get("response") for r in results}) == 1
        )

    def persist_seed(self) -> None:
        self.client.delete("/v1/cache")
        body, calls = self.chat(MISS_Q)
        self.check(
            "persistence seed: entry created (miss)", body.get("cache_hit") is False and calls == 1
        )

    def persist_check(self) -> None:
        stats = self.client.get("/v1/cache/stats").json()
        print(f"      /v1/cache/stats after restart -> {json.dumps(stats)}")
        self.check(
            "persistence: entry survived restart",
            stats.get("entries_count", 0) >= 1,
            f"entries_count={stats.get('entries_count')}",
        )
        body, calls = self.chat(HIT_Q)
        self.check(
            "persistence: reworded request after restart is a HIT without LLM call",
            body.get("cache_hit") is True and calls == 0,
            f"cache_hit={body.get('cache_hit')} llm_calls_delta={calls}",
        )

    def clear(self) -> None:
        removed = self.client.delete("/v1/cache").json()
        entries = self.client.get("/v1/cache/stats").json()["entries_count"]
        self.check(
            "DELETE /v1/cache empties the cache",
            entries == 0,
            f"removed={removed} entries_after={entries}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--phase",
        default="full",
        choices=[
            "full",
            "basic",
            "endpoints",
            "cache",
            "ttl",
            "concurrency",
            "persist-seed",
            "persist-check",
        ],
    )
    parser.add_argument("--ttl-seconds", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    smoke = Smoke(args.url, args.api_key)
    try:
        if args.phase == "full":
            smoke.endpoints()
            smoke.miss_hit_different()
            smoke.ttl(args.ttl_seconds)
            smoke.concurrency(args.concurrency)
            smoke.clear()
        elif args.phase == "basic":
            smoke.endpoints()
            smoke.miss_hit_different()
        elif args.phase == "endpoints":
            smoke.endpoints()
        elif args.phase == "cache":
            smoke.miss_hit_different()
            smoke.clear()
        elif args.phase == "ttl":
            smoke.ttl(args.ttl_seconds)
        elif args.phase == "concurrency":
            smoke.concurrency(args.concurrency)
        elif args.phase == "persist-seed":
            smoke.persist_seed()
        else:
            smoke.persist_check()
    except httpx.HTTPError as exc:
        smoke.check("server reachable", False, f"{type(exc).__name__}: {exc}")
    print(f"RESULT: {'PASS' if smoke.failures == 0 else 'FAIL'} ({smoke.failures} failed checks)")
    sys.exit(0 if smoke.failures == 0 else 1)


if __name__ == "__main__":
    main()
