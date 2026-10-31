"""Local neural embeddings via sentence-transformers (optional extra).

Install with ``pip install -e ".[sentence-transformers]"`` (pulls in PyTorch). The first
run downloads the model from the Hugging Face Hub (all-MiniLM-L6-v2 is ~90 MB) into the
Hugging Face cache (``HF_HOME``); later runs load it from disk and work offline.
"""

from __future__ import annotations

import asyncio
import importlib
from typing import Any

import numpy as np

from llmops_cache.embeddings.base import EmbeddingError, EmbeddingProvider, Vector, l2_normalise


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        try:
            module = importlib.import_module("sentence_transformers")
        except ImportError as exc:
            raise EmbeddingError(
                "EMBEDDING_PROVIDER=sentence-transformers but the package is not installed. "
                'Install it with: python -m pip install -e ".[sentence-transformers]"'
            ) from exc
        try:
            # Pinning the device (CPU by default) keeps vectors reproducible across machines.
            self._model: Any = module.SentenceTransformer(model_name, device=device)
        except Exception as exc:
            raise EmbeddingError(
                f"could not load sentence-transformers model {model_name!r} "
                f"({type(exc).__name__}). The first run downloads it from the Hugging Face "
                "Hub, so it needs network access once; afterwards it is read from the local "
                "Hugging Face cache."
            ) from exc

        dims = self._model.get_sentence_embedding_dimension()
        if not dims:  # some models do not declare it; measure instead
            dims = len(self._encode(["dimension probe"])[0])
        self.dimensions = int(dims)
        self.name = f"st-{model_name}"

    def _encode(self, texts: list[str]) -> list[Vector]:
        matrix = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [l2_normalise(np.asarray(row, dtype=np.float32)) for row in matrix]

    async def embed_batch(self, texts: list[str]) -> list[Vector]:
        if not texts:
            return []
        try:
            # Model inference is CPU bound; keep it off the event loop.
            return await asyncio.to_thread(self._encode, texts)
        except Exception as exc:
            raise EmbeddingError(
                f"sentence-transformers encode failed: {type(exc).__name__}"
            ) from exc
