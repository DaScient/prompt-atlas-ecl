"""Environment-driven embeddings selection.

``PAE_EMBEDDINGS_PROVIDER`` chooses:

* ``hashing`` (default) — pure-Python, dependency-free.
* ``openai``           — requires ``OPENAI_API_KEY`` + ``openai`` SDK.
* ``sentence-transformers`` — requires the ``sentence-transformers`` pkg.

Missing optional deps cleanly fall back to ``hashing``.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from src.embeddings.base import EmbeddingProvider
from src.embeddings.hashing import HashingEmbeddings
from src.embeddings.openai_embed import OpenAIEmbeddings
from src.embeddings.sentence_transformers_embed import SentenceTransformersEmbeddings

logger = logging.getLogger(__name__)


def _fallback(reason: str, *, dim: int = 256) -> EmbeddingProvider:
    logger.info("Embeddings falling back to hashing: %s", reason)
    return HashingEmbeddings(dim=dim)


def get_default_embeddings(
    provider_name: Optional[str] = None,
    *,
    dim: int = 256,
) -> EmbeddingProvider:
    name = (
        provider_name or os.getenv("PAE_EMBEDDINGS_PROVIDER", "hashing")
    ).strip().lower()

    if name == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            return _fallback("OPENAI_API_KEY unset", dim=dim)
        try:
            return OpenAIEmbeddings()
        except Exception as exc:  # pragma: no cover
            return _fallback(f"openai init failed: {exc}", dim=dim)

    if name in {"sentence-transformers", "st"}:
        try:
            return SentenceTransformersEmbeddings()
        except Exception as exc:  # pragma: no cover
            return _fallback(f"sentence-transformers init failed: {exc}", dim=dim)

    if name not in {"hashing", ""}:
        logger.info("Unknown PAE_EMBEDDINGS_PROVIDER=%r; using hashing", name)
    return HashingEmbeddings(dim=dim)


__all__ = ["get_default_embeddings"]
