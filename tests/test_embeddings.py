"""Phase 5 — embedding provider tests."""
import math

import pytest

from src.embeddings import HashingEmbeddings, get_default_embeddings


def _cosine(a, b):
    n = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    return n / (da * db) if da and db else 0.0


def test_hashing_is_deterministic():
    e = HashingEmbeddings(dim=64)
    a = e.embed(["hello world"])[0]
    b = e.embed(["hello world"])[0]
    assert a == b


def test_hashing_dim_is_respected():
    e = HashingEmbeddings(dim=128)
    out = e.embed(["a", "b"])
    assert len(out) == 2
    assert all(len(v) == 128 for v in out)


def test_hashing_vectors_are_unit_norm():
    e = HashingEmbeddings(dim=64)
    v = e.embed(["the quick brown fox jumps over the lazy dog"])[0]
    norm = math.sqrt(sum(x * x for x in v))
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_hashing_empty_string_does_not_crash():
    e = HashingEmbeddings(dim=32)
    out = e.embed(["", "non-empty"])
    assert len(out) == 2
    # Empty string ⇒ all-zero vector (no n-grams contributed).
    assert all(x == 0.0 for x in out[0])


def test_hashing_similar_strings_score_above_unrelated():
    e = HashingEmbeddings(dim=512, ngram=3)
    a, b, c = e.embed([
        "the cat sat on the mat",
        "the cat sat on the rug",
        "quantum entanglement theory",
    ])
    sim_close = _cosine(a, b)
    sim_far = _cosine(a, c)
    # Related sentences should be much closer than unrelated ones.
    assert sim_close > sim_far


def test_hashing_is_case_insensitive():
    e = HashingEmbeddings(dim=64)
    a = e.embed(["Hello World"])[0]
    b = e.embed(["hello world"])[0]
    assert a == b


def test_hashing_rejects_invalid_dim():
    with pytest.raises(ValueError):
        HashingEmbeddings(dim=0)


# -------------------------------------------------------------- factory


def test_default_factory_returns_hashing_when_unset(monkeypatch):
    for var in ("PAE_EMBEDDINGS_PROVIDER", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    p = get_default_embeddings(dim=128)
    assert isinstance(p, HashingEmbeddings)
    assert p.dim == 128


def test_default_factory_openai_without_key_falls_back(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("PAE_EMBEDDINGS_PROVIDER", "openai")
    p = get_default_embeddings()
    assert isinstance(p, HashingEmbeddings)
