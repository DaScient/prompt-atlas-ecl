"""Embedding provider protocol."""
from __future__ import annotations

from typing import List, Protocol, Sequence, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Map a batch of strings to a batch of fixed-length float vectors."""

    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> List[List[float]]: ...


__all__ = ["EmbeddingProvider"]
