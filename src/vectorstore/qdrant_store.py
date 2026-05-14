"""Qdrant-backed memory store with an in-memory fallback.

The fallback is a tiny pure-Python cosine-similarity index — fine for tests,
demos, and CI, and just enough to keep the public ``CoLearningMemoryStore``
API consistent regardless of whether Qdrant is reachable.
"""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional, Tuple

from .schema import COLLECTION_DEFAULT, DEFAULT_VECTOR_SIZE, MemoryPoint, PointPayload

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------
class _VectorBackend(ABC):
    @abstractmethod
    def ensure_collection(self, name: str, vector_size: int) -> None: ...

    @abstractmethod
    def upsert(self, name: str, points: Iterable[MemoryPoint]) -> None: ...

    @abstractmethod
    def search(
        self, name: str, vector: List[float], limit: int
    ) -> List[Tuple[float, MemoryPoint]]: ...


# ---------------------------------------------------------------------------
# In-memory backend (default fallback)
# ---------------------------------------------------------------------------
def _cosine(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        return 0.0
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / (da * db)


class InMemoryVectorBackend(_VectorBackend):
    """Tiny exact-search backend; O(N) per query but dependency-free."""

    def __init__(self) -> None:
        self._collections: dict[str, list[MemoryPoint]] = {}

    def ensure_collection(self, name: str, vector_size: int) -> None:
        self._collections.setdefault(name, [])

    def upsert(self, name: str, points: Iterable[MemoryPoint]) -> None:
        bucket = self._collections.setdefault(name, [])
        index = {p.id: i for i, p in enumerate(bucket)}
        for p in points:
            if p.id in index:
                bucket[index[p.id]] = p
            else:
                index[p.id] = len(bucket)
                bucket.append(p)

    def search(
        self, name: str, vector: List[float], limit: int
    ) -> List[Tuple[float, MemoryPoint]]:
        bucket = self._collections.get(name, [])
        scored = [(_cosine(vector, p.vector), p) for p in bucket]
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored[:limit]


# ---------------------------------------------------------------------------
# Qdrant backend
# ---------------------------------------------------------------------------
class QdrantBackend(_VectorBackend):
    """Production backend. Imports ``qdrant_client`` lazily."""

    def __init__(self, url: str = "http://localhost:6333", api_key: Optional[str] = None):
        try:
            from qdrant_client import QdrantClient  # type: ignore
            from qdrant_client.http import models as qmodels  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "qdrant-client is not installed; install with "
                "`pip install qdrant-client` or use InMemoryVectorBackend"
            ) from exc
        self._client = QdrantClient(url=url, api_key=api_key)
        self._qmodels = qmodels

    def ensure_collection(self, name: str, vector_size: int) -> None:
        qmodels = self._qmodels
        existing = {c.name for c in self._client.get_collections().collections}
        if name in existing:
            return
        self._client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=vector_size, distance=qmodels.Distance.COSINE
            ),
        )

    def upsert(self, name: str, points: Iterable[MemoryPoint]) -> None:
        qmodels = self._qmodels
        batch = [
            qmodels.PointStruct(id=p.id, vector=p.vector, payload=p.payload.model_dump())
            for p in points
        ]
        if batch:
            self._client.upsert(collection_name=name, points=batch)

    def search(
        self, name: str, vector: List[float], limit: int
    ) -> List[Tuple[float, MemoryPoint]]:
        hits = self._client.search(collection_name=name, query_vector=vector, limit=limit)
        out: List[Tuple[float, MemoryPoint]] = []
        for h in hits:
            payload = PointPayload.model_validate(h.payload or {})
            # The Qdrant client does not return the stored vector by default;
            # we surface the *score* + payload, leaving vector empty.
            out.append((float(h.score), MemoryPoint(id=str(h.id), vector=[], payload=payload)))
        return out


# ---------------------------------------------------------------------------
# Public façade
# ---------------------------------------------------------------------------
class CoLearningMemoryStore:
    """Long-term memory for ECL runs.

    Example
    -------
    >>> store = CoLearningMemoryStore()                       # in-memory fallback
    >>> store.remember(run_id="r1", step=3, vector=[0.1]*64,
    ...                e_star=1.4, spec={"steps":[]}, tests=[])
    >>> hits = store.recall([0.1]*64, limit=5)
    """

    def __init__(
        self,
        backend: Optional[_VectorBackend] = None,
        *,
        collection: str = COLLECTION_DEFAULT,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
    ) -> None:
        if backend is None:
            if qdrant_url:
                try:
                    backend = QdrantBackend(url=qdrant_url, api_key=qdrant_api_key)
                except Exception:  # pragma: no cover - exercised when client missing
                    logger.warning(
                        "Qdrant unavailable; falling back to in-memory vector store"
                    )
                    backend = InMemoryVectorBackend()
            else:
                backend = InMemoryVectorBackend()
        self._backend = backend
        self._collection = collection
        self._vector_size = vector_size
        self._backend.ensure_collection(collection, vector_size)

    @property
    def collection(self) -> str:
        return self._collection

    @property
    def backend(self) -> _VectorBackend:
        return self._backend

    def remember(
        self,
        *,
        run_id: str,
        vector: List[float],
        step: int = 0,
        e_star: float = 0.0,
        spec: Optional[dict] = None,
        tests: Optional[List[dict]] = None,
        tags: Optional[List[str]] = None,
        domain: Optional[str] = None,
    ) -> MemoryPoint:
        if len(vector) != self._vector_size:
            raise ValueError(
                f"vector dim {len(vector)} != configured {self._vector_size}"
            )
        point = MemoryPoint(
            vector=list(vector),
            payload=PointPayload(
                run_id=run_id,
                step=step,
                e_star=e_star,
                spec=spec or {},
                tests=tests or [],
                tags=tags or [],
                domain=domain,
            ),
        )
        self._backend.upsert(self._collection, [point])
        return point

    def recall(
        self, vector: List[float], limit: int = 5
    ) -> List[Tuple[float, MemoryPoint]]:
        if len(vector) != self._vector_size:
            raise ValueError(
                f"vector dim {len(vector)} != configured {self._vector_size}"
            )
        return self._backend.search(self._collection, list(vector), limit)
