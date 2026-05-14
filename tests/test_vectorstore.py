"""Tests for the Phase 1 vector-store fallback backend."""
from __future__ import annotations

import pytest

from src.vectorstore import CoLearningMemoryStore, InMemoryVectorBackend


def test_remember_and_recall_returns_top_match() -> None:
    store = CoLearningMemoryStore(backend=InMemoryVectorBackend())
    v1 = [1.0] + [0.0] * 63
    v2 = [0.0, 1.0] + [0.0] * 62

    store.remember(run_id="r1", vector=v1, step=1, e_star=1.5, tags=["a"])
    store.remember(run_id="r2", vector=v2, step=1, e_star=1.7, tags=["b"])

    hits = store.recall(v1, limit=2)
    assert len(hits) == 2
    # v1 is identical → cosine 1.0, must rank first.
    top_score, top_point = hits[0]
    assert top_score == pytest.approx(1.0)
    assert top_point.payload.run_id == "r1"


def test_vector_dim_mismatch_raises() -> None:
    store = CoLearningMemoryStore(backend=InMemoryVectorBackend())
    with pytest.raises(ValueError):
        store.remember(run_id="r1", vector=[0.0, 1.0])  # only 2 dims, expected 64


def test_upsert_overwrites_same_id() -> None:
    backend = InMemoryVectorBackend()
    store = CoLearningMemoryStore(backend=backend)
    v = [1.0] + [0.0] * 63
    p1 = store.remember(run_id="r1", vector=v, e_star=1.0)
    # Reuse the same id by calling backend directly.
    p2 = p1.model_copy(update={"payload": p1.payload.model_copy(update={"e_star": 9.9})})
    backend.upsert(store.collection, [p2])
    hits = store.recall(v, limit=5)
    assert len(hits) == 1  # not duplicated
    assert hits[0][1].payload.e_star == 9.9
