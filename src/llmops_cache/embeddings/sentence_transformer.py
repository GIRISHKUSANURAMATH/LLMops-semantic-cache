"""Local neural embeddings via sentence-transformers (optional extra)."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from llmops_cache.embeddings.base import EmbeddingError, EmbeddingProvider, Vector, l2_normalise


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise EmbeddingError(
                "sentence-transformers is not installed. "
                'Install it with: pip install -e ".[sentence-transformers]"'
            ) from exc
        self._model: Any = SentenceTransformer(model_name)
        dims = self._model.get_sentence_embedding_dimension()
        self.dimensions = int(dims)
        self.name = f"st-{model_name}"

    def _encode(self, texts: list[str]) -> list[Vector]:
        matrix = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [l2_normalise(np.asarray(row)) for row in matrix]

    async def embed_batch(self, texts: list[str]) -> list[Vector]:
        # Model inference is CPU bound; keep it off the event loop.
        return await asyncio.to_thread(self._encode, texts)
