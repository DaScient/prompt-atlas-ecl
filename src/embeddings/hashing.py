"""Hashed-n-gram embeddings — the universal fallback.

A trivial but *honest* embedding: each character n-gram is hashed into a
fixed-dim bucket, with sub-quadratic L2 normalisation at the end. The
output is deterministic, dependency-free, and similar-text-aware enough
that the cosine of two related strings is meaningfully > the cosine of
two unrelated strings — which is all the rest of the stack needs to
treat this as a real embedding stream.

This is the technique used historically in scikit-learn's
``HashingVectorizer`` and Vowpal Wabbit — it's a real method, not a stub.
"""
from __future__ import annotations

import hashlib
import math
from typing import List, Sequence


class HashingEmbeddings:
    """Pure-Python hashed-n-gram embedder."""

    name = "hashing"

    def __init__(self, *, dim: int = 256, ngram: int = 3) -> None:
        if dim <= 0:
            raise ValueError("dim must be > 0")
        if ngram <= 0:
            raise ValueError("ngram must be > 0")
        self.dim = dim
        self._ngram = ngram

    # --------------------------------------------------------------- private

    def _vec(self, text: str) -> List[float]:
        v = [0.0] * self.dim
        if not text:
            return v

        # Pad so short strings still emit at least one n-gram, and
        # lowercase to make the embedder case-insensitive — important
        # because briefs/specs come in mixed case.
        t = text.lower()
        if len(t) < self._ngram:
            t = t.ljust(self._ngram)

        for i in range(len(t) - self._ngram + 1):
            gram = t[i : i + self._ngram].encode("utf-8")
            h = hashlib.blake2b(gram, digest_size=8).digest()
            # 6 bytes → bucket index, 1 bit → sign (signed-hash trick
            # reduces collision bias, à la HashingVectorizer).
            bucket = int.from_bytes(h[:6], "big") % self.dim
            sign = 1.0 if (h[7] & 1) else -1.0
            v[bucket] += sign

        # L2-normalise so cosine similarity is just a dot product later.
        norm = math.sqrt(sum(x * x for x in v))
        if norm > 0:
            v = [x / norm for x in v]
        return v

    # ---------------------------------------------------------------- public

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._vec(t or "") for t in texts]


__all__ = ["HashingEmbeddings"]
