"""Centralised, environment-driven configuration."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime settings. Every field can be overridden with an environment variable."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: Literal["development", "test", "production"] = "development"
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = True
    log_prompts: bool = False

    cache_similarity_threshold: float = Field(default=0.90, gt=0.0, le=1.0)
    cache_ttl_seconds: int = Field(default=3600, ge=1)
    cache_max_entries: int = Field(default=10_000, ge=1)
    cache_top_k: int = Field(default=5, ge=1, le=100)
    cache_cleanup_interval_seconds: int = Field(default=60, ge=0)
    cache_max_temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    vector_store_path: str = ""

    embedding_provider: Literal["hashing", "sentence-transformers", "openai"] = "hashing"
    embedding_dimensions: int = Field(default=512, ge=16, le=8192)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    llm_provider: Literal["mock", "openai"] = "mock"
    llm_default_model: str = "mock-1"
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    mock_llm_latency_ms: int = Field(default=800, ge=0)
    openai_api_key: SecretStr | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_embedding_model: str = "text-embedding-3-small"

    cost_per_1k_input_tokens: float = Field(default=0.00015, ge=0)
    cost_per_1k_output_tokens: float = Field(default=0.0006, ge=0)

    max_message_chars: int = Field(default=8000, ge=1)
    max_request_bytes: int = Field(default=64_000, ge=1024)
    cors_allow_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    api_keys: Annotated[list[SecretStr], NoDecode] = Field(default_factory=list)

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("api_keys", mode="before")
    @classmethod
    def _split_keys(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if "*" in self.cors_allow_origins and self.app_env == "production":
            raise ValueError("CORS_ALLOW_ORIGINS='*' is not allowed when APP_ENV=production")
        needs_key = self.llm_provider == "openai" or self.embedding_provider == "openai"
        if needs_key and not (self.openai_api_key and self.openai_api_key.get_secret_value()):
            raise ValueError("OPENAI_API_KEY is required when an OpenAI provider is selected")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
