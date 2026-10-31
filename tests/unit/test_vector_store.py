from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from llmops_cache.cache.entry import CacheEntry
from llmops_cache.cache.vector_store import InMemoryVectorStore, VectorStoreError


def entry(entry_id: str, ctx: str = "ctx", created: float = 0.0, ttl: int = 100) -> CacheEntry:
    return CacheEntry(
        id=entry_id,
        context_key=ctx,
        query=entry_id,
        normalized_query=entry_id,
        response=f"answer {entry_id}",
        provider="mock",
        model="m",
        embedding_model="emb",
        created_at=created,
        expires_at=created + ttl,
        ttl_seconds=ttl,
        input_tokens=1,
        output_tokens=2,
    )


def unit(*values: float) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_search_orders_by_similarity() -> None:
    store = InMemoryVectorStore()
    store.add(entry("x"), unit(1, 0, 0))
    store.add(entry("y"), unit(1, 1, 0))
    store.add(entry("z"), unit(0, 0, 1))
    hits = store.search(unit(1, 0.1, 0), "ctx", top_k=3)
    assert [h.entry.id for h in hits] == ["x", "y", "z"]
    assert hits[0].similarity > hits[1].similarity > hits[2].similarity


def test_search_is_partitioned_by_context_key() -> None:
    store = InMemoryVectorStore()
    store.add(entry("a", ctx="model-1"), unit(1, 0))
    assert store.search(unit(1, 0), "model-2", top_k=5) == []
    assert len(store.search(unit(1, 0), "model-1", top_k=5)) == 1


def test_top_k_limits_results() -> None:
    store = InMemoryVectorStore()
    for i in range(10):
        store.add(entry(str(i)), unit(1, i))
    assert len(store.search(unit(1, 0), "ctx", top_k=3)) == 3


def test_dimension_mismatch_raises() -> None:
    store = InMemoryVectorStore()
    store.add(entry("a"), unit(1, 0))
    with pytest.raises(VectorStoreError):
        store.add(entry("b"), unit(1, 0, 0))
    with pytest.raises(VectorStoreError):
        store.search(unit(1, 0, 0), "ctx", top_k=1)


def test_delete_keeps_remaining_rows_aligned() -> None:
    store = InMemoryVectorStore()
    store.add(entry("a"), unit(1, 0))
    store.add(entry("b"), unit(0, 1))
    store.add(entry("c"), unit(-1, 0))
    assert store.delete("b") is True
    assert store.delete("missing") is False
    hit = store.search(unit(-1, 0), "ctx", top_k=1)[0]
    assert hit.entry.id == "c"
    assert hit.similarity == pytest.approx(1.0)


def test_clear_and_purge() -> None:
    store = InMemoryVectorStore()
    store.add(entry("old", created=0, ttl=10), unit(1, 0))
    store.add(entry("new", created=5, ttl=100), unit(0, 1))
    assert store.purge_expired(now=20) == 1
    assert [e.id for e in store.entries()] == ["new"]
    assert store.clear() == 1
    assert len(store) == 0


def test_lru_eviction_at_capacity() -> None:
    store = InMemoryVectorStore(max_entries=2)
    a, b = entry("a", created=1), entry("b", created=2)
    store.add(a, unit(1, 0))
    store.add(b, unit(0, 1))
    a.record_hit(now=10, similarity=1.0)  # 'a' is now the most recently used
    store.add(entry("c", created=3), unit(1, 1))
    assert {e.id for e in store.entries()} == {"a", "c"}


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    store = InMemoryVectorStore()
    store.add(entry("a"), unit(1, 0))
    store.add(entry("b", ctx="other"), unit(0, 1))
    store.save(tmp_path, embedding_model="emb")

    restored = InMemoryVectorStore()
    assert restored.load(tmp_path, embedding_model="emb", now=1) == 2
    assert restored.search(unit(1, 0), "ctx", top_k=1)[0].entry.response == "answer a"


def test_load_skips_expired_and_incompatible_indexes(tmp_path: Path) -> None:
    store = InMemoryVectorStore()
    store.add(entry("a", ttl=10), unit(1, 0))
    store.save(tmp_path, embedding_model="emb")
    assert InMemoryVectorStore().load(tmp_path, "emb", now=50) == 0  # expired
    assert InMemoryVectorStore().load(tmp_path, "different-model", now=1) == 0


def test_load_refuses_pickled_payloads(tmp_path: Path) -> None:
    """A tampered vectors.npy containing pickled objects must not be deserialised."""
    store = InMemoryVectorStore()
    store.add(entry("a"), unit(1, 0))
    store.save(tmp_path, embedding_model="emb")
    np.save(tmp_path / "vectors.npy", np.array([{"evil": 1}], dtype=object), allow_pickle=True)
    assert InMemoryVectorStore().load(tmp_path, "emb", now=1) == 0


def test_load_handles_corrupt_metadata(tmp_path: Path) -> None:
    (tmp_path / "entries.json").write_text("{not json", encoding="utf-8")
    np.save(tmp_path / "vectors.npy", np.zeros((0, 2), dtype=np.float32))
    assert InMemoryVectorStore().load(tmp_path, "emb", now=1) == 0
    (tmp_path / "entries.json").write_text(json.dumps({"format_version": 99}), encoding="utf-8")
    assert InMemoryVectorStore().load(tmp_path, "emb", now=1) == 0
