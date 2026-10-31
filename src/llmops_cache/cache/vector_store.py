"""Vector store interface and an in-process NumPy implementation.

The NumPy store performs exact (brute-force) cosine search, partitioned by ``context_key``.
That is O(n) per namespace per lookup, which is fast enough for tens of thousands of entries
and has zero setup cost. Larger deployments should implement :class:`VectorStore` on top of
FAISS, Qdrant, pgvector, Redis etc.; the cache service only depends on the interface.

Persistence uses JSON + ``.npy`` loaded with ``allow_pickle=False`` (no unsafe deserialisation).
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from llmops_cache.cache.entry import CacheEntry
from llmops_cache.embeddings.base import Vector

logger = logging.getLogger(__name__)

_FORMAT_VERSION = 1


class VectorStoreError(RuntimeError):
    """The vector store could not complete an operation."""


@dataclass(frozen=True, slots=True)
class SearchHit:
    entry: CacheEntry
    similarity: float


class VectorStore(ABC):
    @abstractmethod
    def add(self, entry: CacheEntry, vector: Vector) -> None: ...

    @abstractmethod
    def search(self, vector: Vector, context_key: str, top_k: int) -> list[SearchHit]:
        """Return up to ``top_k`` entries of ``context_key`` ordered by descending similarity."""

    @abstractmethod
    def get(self, entry_id: str) -> CacheEntry | None: ...

    @abstractmethod
    def delete(self, entry_id: str) -> bool: ...

    @abstractmethod
    def clear(self) -> int: ...

    @abstractmethod
    def purge_expired(self, now: float) -> int: ...

    @abstractmethod
    def entries(self) -> list[CacheEntry]: ...

    @abstractmethod
    def __len__(self) -> int: ...


@dataclass
class _Namespace:
    ids: list[str] = field(default_factory=list)
    matrix: np.ndarray | None = None


class InMemoryVectorStore(VectorStore):
    """Exact cosine search over unit vectors with LRU eviction at ``max_entries``."""

    def __init__(self, max_entries: int = 10_000) -> None:
        self.max_entries = max_entries
        self.dimensions: int | None = None
        self._entries: dict[str, CacheEntry] = {}
        self._namespace_of: dict[str, str] = {}
        self._namespaces: dict[str, _Namespace] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, entry: CacheEntry, vector: Vector) -> None:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if self.dimensions is None:
            self.dimensions = int(vector.shape[0])
        if vector.shape[0] != self.dimensions:
            raise VectorStoreError(
                f"vector has {vector.shape[0]} dimensions, store expects {self.dimensions}"
            )
        if entry.id in self._entries:
            self.delete(entry.id)
        while len(self._entries) >= self.max_entries:
            self._evict_one()

        ns = self._namespaces.setdefault(entry.context_key, _Namespace())
        row = vector[np.newaxis, :]
        ns.matrix = row if ns.matrix is None else np.vstack([ns.matrix, row])
        ns.ids.append(entry.id)
        self._entries[entry.id] = entry
        self._namespace_of[entry.id] = entry.context_key

    def search(self, vector: Vector, context_key: str, top_k: int) -> list[SearchHit]:
        ns = self._namespaces.get(context_key)
        if ns is None or ns.matrix is None or not ns.ids:
            return []
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != ns.matrix.shape[1]:
            raise VectorStoreError("query vector dimension does not match the index")
        scores = ns.matrix @ query
        k = min(top_k, len(ns.ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [
            SearchHit(self._entries[ns.ids[i]], float(np.clip(scores[i], -1.0, 1.0))) for i in top
        ]

    def get(self, entry_id: str) -> CacheEntry | None:
        return self._entries.get(entry_id)

    def delete(self, entry_id: str) -> bool:
        key = self._namespace_of.pop(entry_id, None)
        if key is None:
            return False
        del self._entries[entry_id]
        ns = self._namespaces[key]
        index = ns.ids.index(entry_id)
        ns.ids.pop(index)
        if ns.ids and ns.matrix is not None:
            ns.matrix = np.delete(ns.matrix, index, axis=0)
        else:
            del self._namespaces[key]
        return True

    def clear(self) -> int:
        count = len(self._entries)
        self._entries.clear()
        self._namespace_of.clear()
        self._namespaces.clear()
        return count

    def purge_expired(self, now: float) -> int:
        expired = [e.id for e in self._entries.values() if e.is_expired(now)]
        for entry_id in expired:
            self.delete(entry_id)
        return len(expired)

    def entries(self) -> list[CacheEntry]:
        return list(self._entries.values())

    def _evict_one(self) -> None:
        def recency(entry: CacheEntry) -> float:
            return entry.last_accessed_at or entry.created_at

        victim = min(self._entries.values(), key=recency)
        self.delete(victim.id)

    # ----------------------------------------------------------------- persistence
    def save(self, directory: Path, embedding_model: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        ordered = list(self._entries.values())
        if ordered:
            vectors = np.vstack([self._vector_of(e.id) for e in ordered]).astype(np.float32)
        else:
            vectors = np.zeros((0, self.dimensions or 0), dtype=np.float32)
        meta = {
            "format_version": _FORMAT_VERSION,
            "embedding_model": embedding_model,
            "dimensions": self.dimensions,
            "entries": [e.to_dict() for e in ordered],
        }
        _atomic_write_bytes(directory / "entries.json", json.dumps(meta).encode("utf-8"))
        tmp = directory / "vectors.npy.tmp"
        with tmp.open("wb") as handle:
            np.save(handle, vectors, allow_pickle=False)
        os.replace(tmp, directory / "vectors.npy")

    def load(self, directory: Path, embedding_model: str, now: float) -> int:
        meta_path, vec_path = directory / "entries.json", directory / "vectors.npy"
        if not meta_path.exists() or not vec_path.exists():
            return 0
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            vectors = np.load(vec_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            logger.warning("cache.load_failed", extra={"error": type(exc).__name__})
            return 0
        if meta.get("format_version") != _FORMAT_VERSION:
            logger.warning("cache.load_skipped", extra={"reason": "format_version"})
            return 0
        if meta.get("embedding_model") != embedding_model:
            logger.warning("cache.load_skipped", extra={"reason": "embedding_model_changed"})
            return 0
        entries = [CacheEntry.from_dict(item) for item in meta.get("entries", [])]
        if len(entries) != len(vectors):
            logger.warning("cache.load_skipped", extra={"reason": "corrupt_index"})
            return 0
        loaded = 0
        for entry, vector in zip(entries, vectors, strict=True):
            if not entry.is_expired(now):
                self.add(entry, vector)
                loaded += 1
        return loaded

    def _vector_of(self, entry_id: str) -> np.ndarray:
        ns = self._namespaces[self._namespace_of[entry_id]]
        assert ns.matrix is not None  # noqa: S101 - invariant: namespace with ids has a matrix
        row: np.ndarray = ns.matrix[ns.ids.index(entry_id)]
        return row


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
