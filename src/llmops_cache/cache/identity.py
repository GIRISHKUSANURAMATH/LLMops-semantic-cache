"""Query normalisation and cache-identity (context key) computation.

Two requests may only share a cached response when *everything that shapes the answer*
other than the user's wording is identical. That set is hashed into a ``context_key`` and
similarity search is restricted to entries with the same key.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[\s?!.;:,]+$")


def normalize_query(text: str) -> str:
    """Canonical form used for embedding and exact-match coalescing."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return _TRAILING_PUNCT_RE.sub("", text)


def context_key(
    *,
    provider: str,
    model: str,
    system_prompt: str | None,
    temperature: float,
    max_tokens: int,
    namespace: str | None,
    embedding: str = "",
) -> str:
    material = {
        "provider": provider,
        "embedding": embedding,
        "model": model,
        "system_prompt": system_prompt or "",
        "temperature": round(temperature, 3),
        "max_tokens": max_tokens,
        "namespace": namespace or "",
    }
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fingerprint(text: str) -> str:
    """Short, non-reversible identifier for logging a prompt without logging its content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
