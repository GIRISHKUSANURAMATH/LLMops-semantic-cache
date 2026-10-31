"""Command-line client for a running server.

    python -m llmops_cache.cli health
    python -m llmops_cache.cli chat "What is machine learning?"
    python -m llmops_cache.cli stats | metrics | clear-cache | demo

Reads LLMOPS_URL (default http://127.0.0.1:8000) and LLMOPS_API_KEY from the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import httpx

DEMO_MESSAGES = (
    "What is machine learning?",
    "Explain machine learning to me.",
    "What is the capital of France?",
)


def _client(url: str) -> httpx.Client:
    headers = {}
    api_key = os.environ.get("LLMOPS_API_KEY")
    if api_key:
        headers["X-API-Key"] = api_key
    return httpx.Client(base_url=url.rstrip("/"), headers=headers, timeout=60)


def _print(data: Any) -> None:
    print(json.dumps(data, indent=2))


def _call(client: httpx.Client, method: str, path: str, **kwargs: Any) -> Any:
    response = client.request(method, path, **kwargs)
    if response.status_code >= 400:
        print(f"HTTP {response.status_code}: {response.text}", file=sys.stderr)
        raise SystemExit(1)
    return response.json()


def _demo(client: httpx.Client) -> None:
    for message in DEMO_MESSAGES:
        body = _call(client, "POST", "/v1/chat", json={"message": message})
        status = "CACHE HIT " if body["cache_hit"] else "CACHE MISS"
        similarity = body["similarity"] if body["similarity"] is not None else "-"
        print(
            f"{status} source={body['source']:<14} similarity={similarity!s:<6} "
            f"latency={body['latency_ms']:.1f}ms  {message!r}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="llmops-cache", description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=os.environ.get("LLMOPS_URL", "http://127.0.0.1:8000"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("health")
    sub.add_parser("stats")
    sub.add_parser("metrics")
    sub.add_parser("clear-cache")
    sub.add_parser("demo", help="send a MISS / HIT / MISS sequence to the server")
    chat = sub.add_parser("chat")
    chat.add_argument("message")
    chat.add_argument("--model", default="default")
    chat.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)

    try:
        with _client(args.url) as client:
            if args.command == "health":
                _print(_call(client, "GET", "/health"))
            elif args.command == "stats":
                _print(_call(client, "GET", "/v1/cache/stats"))
            elif args.command == "metrics":
                _print(_call(client, "GET", "/v1/metrics"))
            elif args.command == "clear-cache":
                _print(_call(client, "DELETE", "/v1/cache"))
            elif args.command == "demo":
                _demo(client)
            else:
                payload = {
                    "message": args.message,
                    "model": args.model,
                    "use_cache": not args.no_cache,
                }
                _print(_call(client, "POST", "/v1/chat", json=payload))
    except httpx.ConnectError:
        print(f"Cannot reach {args.url}. Is the server running?", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
