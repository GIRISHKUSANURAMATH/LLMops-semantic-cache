"""Cache entry model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class CacheEntry:
    """One cached LLM response. The embedding itself lives in the vector store, keyed by ``id``.

    Timestamps are Unix epoch seconds (float) so TTL checks are cheap and timezone-free.
    """

    id: str
    context_key: str
    query: str
    normalized_query: str
    response: str
    provider: str
    model: str
    embedding_model: str
    created_at: float
    expires_at: float
    ttl_seconds: int
    input_tokens: int
    output_tokens: int
    hit_count: int = 0
    last_accessed_at: float | None = None
    last_hit_similarity: float | None = None

    def is_expired(self, now: float) -> bool:
        return now >= self.expires_at

    def record_hit(self, now: float, similarity: float) -> None:
        self.hit_count += 1
        self.last_accessed_at = now
        self.last_hit_similarity = similarity

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheEntry:
        return cls(**data)
