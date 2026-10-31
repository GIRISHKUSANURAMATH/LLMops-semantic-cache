"""Public request/response schemas (what appears in /docs)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_NAMESPACE_PATTERN = r"^[A-Za-z0-9_.:-]{1,64}$"


class ChatRequestBody(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"message": "Explain machine learning", "model": "default"}]
        },
    )

    message: str = Field(min_length=1, description="The user prompt.")
    model: str = Field(
        default="default",
        min_length=1,
        max_length=100,
        description='Upstream model name. "default" uses LLM_DEFAULT_MODEL.',
    )
    system_prompt: str | None = Field(default=None, max_length=8000)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=32_000)
    namespace: str | None = Field(
        default=None,
        pattern=_NAMESPACE_PATTERN,
        description="Isolates cache entries, e.g. per tenant or per prompt-template version.",
    )
    use_cache: bool = Field(default=True, description="Set false to force an LLM call.")
    ttl_seconds: int | None = Field(default=None, ge=1, le=30 * 24 * 3600)

    @field_validator("message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class ChatResponseBody(BaseModel):
    response: str
    cache_hit: bool
    source: Literal["llm", "semantic_cache", "coalesced"]
    similarity: float | None = Field(description="Cosine similarity of the matched entry.")
    latency_ms: float
    model: str
    cache_entry_id: str | None
    usage: Usage
    estimated_cost_saved_usd: float = Field(description="Estimate, not billing data.")
    cache_bypass_reason: str | None = None


class CacheSearchBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1)
    model: str = Field(default="default", min_length=1, max_length=100)
    system_prompt: str | None = Field(default=None, max_length=8000)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=32_000)
    namespace: str | None = Field(default=None, pattern=_NAMESPACE_PATTERN)
    threshold: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: int = Field(default=5, ge=1, le=50)


class CacheSearchMatch(BaseModel):
    entry_id: str
    query: str
    similarity: float
    would_hit: bool
    expired: bool
    hit_count: int
    created_at: float
    expires_at: float


class CacheSearchResponse(BaseModel):
    threshold: float
    matches: list[CacheSearchMatch]


class CacheStats(BaseModel):
    entries_count: int
    max_entries: int
    expired_entries: int
    total_hits_served: int
    similarity_threshold: float
    ttl_seconds: int
    embedding_provider: str
    embedding_dimensions: int
    llm_provider: str


class ClearResponse(BaseModel):
    removed: int


class MetricsResponse(BaseModel):
    requests_total: int
    cache_hits: int
    cache_misses: int
    coalesced_requests: int
    cache_bypassed: int
    errors: int
    hit_rate: float
    llm_calls: int
    llm_calls_avoided: int
    estimated_tokens_avoided: int
    estimated_cost_saved_usd: float
    average_latency_ms: float | None
    average_hit_latency_ms: float | None
    average_miss_latency_ms: float | None
    entries_count: int
    note: str = "Cost and token figures are estimates based on configured per-token prices."


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    llm_provider: str
    embedding_provider: str
    entries_count: int


class ErrorBody(BaseModel):
    error: str
    message: str
    request_id: str | None = None
