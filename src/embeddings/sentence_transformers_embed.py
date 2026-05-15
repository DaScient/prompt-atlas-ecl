"""Local sentence-transformers embeddings — lazy import.

Useful when running fully self-hosted (no cloud API key) but still
wanting genuine semantic embeddings. The model name defaults to
``all-MiniLM-L6-v2`` (384-dim) — fast and CPU-friendly.
"""
from __future__ import annotations

import logging
import os
from typing import List, Sequence

logger = logging.getLogger(__name__)


class SentenceTransformersEmbeddings:
    name = "sentence-transformers"

    def __init__(self, *, model: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = os.getenv("PAE_ST_EMBED_MODEL", model)
        self._model = None
        # The actual dim is filled in once the model loads.
        self.dim = 384

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers not installed; "
                "`pip install sentence-transformers` or use HashingEmbeddings"
            ) from exc
        self._model = SentenceTransformer(self._model_name)
        # ``get_sentence_embedding_dimension`` is the canonical accessor.
        try:
            self.dim = int(self._model.get_sentence_embedding_dimension())
        except Exception:  # pragma: no cover
            pass
        return self._model

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        model = self._ensure_model()
        # ``convert_to_numpy=True`` is the default; tolist() keeps the
        # public contract pure-Python.
        arr = model.encode(list(texts), normalize_embeddings=True)
        return [list(map(float, row)) for row in arr]


__all__ = ["SentenceTransformersEmbeddings"]
