"""Embedding provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.float32]


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider cannot produce a vector."""


class EmbeddingProvider(ABC):
    """Turns text into L2-normalised float32 vectors so dot product == cosine similarity."""

    #: Stable identifier persisted with the index; changing it invalidates stored vectors.
    name: str
    dimensions: int

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[Vector]:
        """Embed several texts. Implementations must return unit-length vectors."""

    async def embed_text(self, text: str) -> Vector:
        vectors = await self.embed_batch([text])
        return vectors[0]

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release resources (HTTP clients etc.)."""


def l2_normalise(vector: npt.NDArray[np.floating]) -> Vector:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)
