"""Deterministic, dependency-free embedding provider.

This is a *lexical* embedding (feature hashing over stemmed content words, word bigrams and
character trigrams). It needs no model download, no network and no GPU, which makes it the
right default for tests, CI and demos. It captures paraphrases that share content words
("What is machine learning?" / "Explain machine learning to me") but NOT true synonyms
("car" vs "automobile"). Use the sentence-transformers or OpenAI provider for real semantics.
"""

from __future__ import annotations

import hashlib
import itertools
import re

import numpy as np

from llmops_cache.embeddings.base import EmbeddingProvider, Vector, l2_normalise

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Function words plus "question framing" words that rarely change what is being asked.
_STOPWORDS = frozenset(
    """
    a an the is are was were be been being am do does did doing have has had of to in on at
    for from by with about as into over under and or but if then so than that this these those
    it its i me my we our you your he she they them their what which who whom whose how why
    when where can could would should will shall may might must please tell explain describe
    define give show let us know want like just some any there here also really kindly briefly
    simple simply terms mean means meaning
    """.split()  # noqa: SIM905 - readable word list
)

_SUFFIXES = (
    "ization",
    "ations",
    "ation",
    "ingly",
    "ness",
    "ment",
    "ings",
    "ing",
    "ies",
    "ied",
    "ed",
    "es",
    "ly",
    "s",
)

_WORD_WEIGHT = 1.0
_BIGRAM_WEIGHT = 0.5
_TRIGRAM_WEIGHT = 0.25


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if len(word) - len(suffix) >= 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def content_tokens(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    kept = [stem for t in tokens if t not in _STOPWORDS and (stem := _stem(t)) not in _STOPWORDS]
    # A query made only of stopwords ("what is it?") still needs a non-zero vector.
    return kept or [_stem(t) for t in tokens]


class HashingEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimensions: int = 512) -> None:
        self.dimensions = dimensions
        self.name = f"hashing-v1-{dimensions}"

    def _index_and_sign(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        return value % self.dimensions, 1.0 if (value >> 63) & 1 else -1.0

    def _add(self, vector: np.ndarray, feature: str, weight: float) -> None:
        index, sign = self._index_and_sign(feature)
        vector[index] += sign * weight

    def _embed_one(self, text: str) -> Vector:
        vector = np.zeros(self.dimensions, dtype=np.float64)
        tokens = content_tokens(text)
        for token in tokens:
            self._add(vector, f"w:{token}", _WORD_WEIGHT)
            padded = f"<{token}>"
            for i in range(len(padded) - 2):
                self._add(vector, f"c:{padded[i : i + 3]}", _TRIGRAM_WEIGHT)
        for left, right in itertools.pairwise(tokens):
            self._add(vector, f"b:{left}_{right}", _BIGRAM_WEIGHT)
        return l2_normalise(vector)

    async def embed_batch(self, texts: list[str]) -> list[Vector]:
        return [self._embed_one(text) for text in texts]
