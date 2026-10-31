"""Provider for any OpenAI-compatible ``/chat/completions`` endpoint.

Works with OpenAI itself and with compatible servers (Groq, Together, vLLM, Ollama's
``/v1`` endpoint...) by changing ``OPENAI_BASE_URL``.
"""

from __future__ import annotations

from typing import Any

import httpx

from llmops_cache.providers.base import (
    GenerationRequest,
    GenerationResult,
    LLMProvider,
    ProviderError,
    estimate_tokens,
)


class OpenAICompatibleProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.message})
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        try:
            response = await self._client.post("/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderError("LLM provider timed out") from exc
        except httpx.HTTPError as exc:
            raise ProviderError("LLM provider unreachable") from exc

        if response.status_code >= 400:
            retryable = response.status_code == 429 or response.status_code >= 500
            raise ProviderError(
                f"LLM provider returned HTTP {response.status_code}", retryable=retryable
            )
        try:
            body = response.json()
            text = body["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError("LLM provider returned an unexpected payload") from exc

        usage = body.get("usage") or {}
        return GenerationResult(
            text=text,
            model=str(body.get("model", request.model)),
            input_tokens=int(usage.get("prompt_tokens") or estimate_tokens(request.message)),
            output_tokens=int(usage.get("completion_tokens") or estimate_tokens(text)),
        )

    async def aclose(self) -> None:
        await self._client.aclose()
