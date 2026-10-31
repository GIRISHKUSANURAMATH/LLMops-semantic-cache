"""LLM provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class ProviderError(RuntimeError):
    """The upstream LLM failed or was unreachable. Mapped to HTTP 502/503."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    message: str
    model: str
    system_prompt: str | None = None
    temperature: float = 0.0
    max_tokens: int = 512


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    metadata: dict[str, str] = field(default_factory=dict)


class LLMProvider(ABC):
    name: str

    @property
    def cache_identity(self) -> str:
        """What must match for two calls to be interchangeable (provider + endpoint)."""
        return self.name

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Return a full completion. Streaming is intentionally not part of this interface yet."""

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        """Release resources."""


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) used when a provider reports no usage."""
    return max(1, (len(text) + 3) // 4)
