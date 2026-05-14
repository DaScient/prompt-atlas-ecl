"""
Vector-store integration for long-term ECL co-learning memory.

The :class:`CoLearningMemoryStore` provides a small, well-typed surface for
persisting "what happened in a run" (final spec/tests, E*, state vector
snapshot, tags) and recalling it later via similarity search. The default
backend is **Qdrant**; when ``qdrant-client`` is not installed or no Qdrant
endpoint is reachable, an in-memory cosine-similarity backend keeps the rest
of the system working unchanged.
"""

from .schema import (
    COLLECTION_DEFAULT,
    DEFAULT_VECTOR_SIZE,
    MemoryPoint,
    PointPayload,
)
from .qdrant_store import CoLearningMemoryStore, InMemoryVectorBackend, QdrantBackend

__all__ = [
    "COLLECTION_DEFAULT",
    "DEFAULT_VECTOR_SIZE",
    "MemoryPoint",
    "PointPayload",
    "CoLearningMemoryStore",
    "InMemoryVectorBackend",
    "QdrantBackend",
]
