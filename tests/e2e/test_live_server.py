"""End-to-end: start the real server process with uvicorn and talk to it over TCP.

Also runs the demo, the CLI and the evaluation scripts exactly as a user would.
Everything uses the mock LLM and the local hashing embedder: no network, no API keys.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _env(**extra: str) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in {"OPENAI_API_KEY", "API_KEYS", "LLM_PROVIDER", "EMBEDDING_PROVIDER"}
    }
    env.update(
        {
            "LLM_PROVIDER": "mock",
            "MOCK_LLM_LATENCY_MS": "200",
            "LOG_LEVEL": "WARNING",
            "VECTOR_STORE_PATH": "",
            "PYTHONIOENCODING": "utf-8",
        }
    )
    env.update(extra)
    return env


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    port = _free_port()
    workdir = tmp_path_factory.mktemp("server")
    # Server output goes to a file, never an unread pipe: a full pipe buffer (small on
    # Windows) would block the server and hang the test run.
    log_path = workdir / "server.log"
    log_file = log_path.open("wb")
    proc = subprocess.Popen(
        [sys.executable, "-m", "llmops_cache"],
        cwd=workdir,
        env=_env(PORT=str(port), HOST="127.0.0.1"),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.2)
    else:
        proc.kill()
        proc.wait(timeout=10)
        log_file.close()
        pytest.fail(f"server did not start:\n{log_path.read_text(errors='replace')}")
    yield url
    proc.terminate()
    proc.wait(timeout=10)
    log_file.close()


def test_full_miss_hit_flow_over_http(server: str) -> None:
    with httpx.Client(base_url=server, timeout=10) as client:
        assert client.delete("/v1/cache").status_code == 200
        miss = client.post("/v1/chat", json={"message": "What is machine learning?"}).json()
        hit = client.post("/v1/chat", json={"message": "Explain machine learning to me."}).json()
        other = client.post("/v1/chat", json={"message": "What is the capital of France?"}).json()

        assert miss["cache_hit"] is False and miss["latency_ms"] >= 200
        assert hit["cache_hit"] is True and hit["source"] == "semantic_cache"
        assert hit["latency_ms"] < miss["latency_ms"] / 10
        assert other["cache_hit"] is False

        stats = client.get("/v1/cache/stats").json()
        assert stats["entries_count"] == 2
        metrics = client.get("/v1/metrics").json()
        assert metrics["cache_hits"] >= 1
        assert client.delete("/v1/cache").json()["removed"] == 2
        assert client.get("/docs").status_code == 200


def test_concurrent_identical_requests_over_http(server: str) -> None:
    import concurrent.futures

    with httpx.Client(base_url=server, timeout=10) as client:
        client.delete("/v1/cache")
        before = client.get("/v1/metrics").json()["llm_calls"]

        def send(_: int) -> dict[str, object]:
            response = client.post("/v1/chat", json={"message": "concurrency test question"})
            return dict(response.json())

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(send, range(8)))
        after = client.get("/v1/metrics").json()["llm_calls"]
        assert after - before == 1
        assert len({r["response"] for r in results}) == 1


def test_cli_against_live_server(server: str) -> None:
    def cli(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603
            [sys.executable, "-m", "llmops_cache.cli", "--url", server, *args],
            capture_output=True,
            text=True,
            env=_env(),
            timeout=60,
            check=False,
        )

    assert '"status": "ok"' in cli("health").stdout
    assert cli("clear-cache").returncode == 0
    demo = cli("demo").stdout.splitlines()
    assert [line.split()[0:2] for line in demo] == [
        ["CACHE", "MISS"],
        ["CACHE", "HIT"],
        ["CACHE", "MISS"],
    ]
    assert '"entries_count": 2' in cli("stats").stdout


def test_cli_reports_unreachable_server() -> None:
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "llmops_cache.cli",
            "--url",
            f"http://127.0.0.1:{_free_port()}",
            "health",
        ],
        capture_output=True,
        text=True,
        env=_env(),
        timeout=30,
        check=False,
    )
    assert result.returncode == 2
    assert "Is the server running?" in result.stderr


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        (["examples/demo.py", "--latency-ms", "20"], ["CACHE MISS", "CACHE HIT", "LLM NOT called"]),
        (["scripts/evaluate_threshold.py"], ["| 0.90 |", "recall"]),
        (
            ["scripts/benchmark.py", "--requests", "40", "--latency-ms", "1"],
            ["overall cache hit rate (includes exact repeats)", "reworded hits"],
        ),
    ],
)
def test_scripts_run_from_repo_root(script: list[str], expected: list[str]) -> None:
    result = subprocess.run(  # noqa: S603
        [sys.executable, *script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=_env(),
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for text in expected:
        assert text in result.stdout


def test_smoke_test_script_passes_against_live_server(server: str) -> None:
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "scripts/smoke_test.py",
            "--url",
            server,
            "--phase",
            "full",
            "--ttl-seconds",
            "1",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=_env(),
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESULT: PASS" in result.stdout
    assert "FAIL " not in result.stdout
