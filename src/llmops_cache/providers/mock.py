"""Deterministic mock LLM: no network, no cost, configurable latency."""

from __future__ import annotations

import asyncio
import hashlib

from llmops_cache.providers.base import (
    GenerationRequest,
    GenerationResult,
    LLMProvider,
    ProviderError,
    estimate_tokens,
)


class MockLLMProvider(LLMProvider):
    """Returns a stable answer derived from the prompt, after an artificial delay.

    ``calls`` counts invocations so tests and the demo can prove whether the LLM was hit.
    Set ``fail_with`` to simulate an upstream outage.
    """

    name = "mock"

    def __init__(self, latency_ms: int = 800) -> None:
        self.latency_ms = latency_ms
        self.calls = 0
        self.fail_with: ProviderError | None = None

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000)
        if self.fail_with is not None:
            raise self.fail_with
        fingerprint = hashlib.sha256(request.message.encode("utf-8")).hexdigest()[:8]
        text = (
            f"[mock:{request.model}] This is a simulated answer to: {request.message.strip()} "
            f"(response id {fingerprint})"
        )
        return GenerationResult(
            text=text,
            model=request.model,
            input_tokens=estimate_tokens((request.system_prompt or "") + request.message),
            output_tokens=estimate_tokens(text),
        )
