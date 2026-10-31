"""OpenAI-compatible embeddings endpoint (``POST {base_url}/embeddings``)."""

from __future__ import annotations

import httpx
import numpy as np

from llmops_cache.embeddings.base import EmbeddingError, EmbeddingProvider, Vector, l2_normalise

_KNOWN_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.dimensions = _KNOWN_DIMENSIONS.get(model, 1536)
        self.name = f"openai-{model}"
        self._model = model
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    async def embed_batch(self, texts: list[str]) -> list[Vector]:
        try:
            response = await self._client.post(
                "/embeddings", json={"model": self._model, "input": texts}
            )
            response.raise_for_status()
            data = response.json()["data"]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise EmbeddingError(f"embedding request failed: {type(exc).__name__}") from exc
        rows = sorted(data, key=lambda item: item["index"])
        vectors = [l2_normalise(np.asarray(row["embedding"], dtype=np.float32)) for row in rows]
        if vectors:
            self.dimensions = int(vectors[0].shape[0])
        return vectors

    async def aclose(self) -> None:
        await self._client.aclose()
