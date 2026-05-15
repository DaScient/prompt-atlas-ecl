"""Phase 5 — Pluggable embedding provider.

The Phase 5 roadmap calls for "Real Embedding Streams" — i.e. moving
beyond the random ``torch.randn`` features in ``Core.step``. This module
exposes a small ``EmbeddingProvider`` protocol with three backends:

* :class:`OpenAIEmbeddings` — lazy import of the official SDK.
* :class:`SentenceTransformersEmbeddings` — lazy import of
  ``sentence_transformers`` for self-hosted local models.
* :class:`HashingEmbeddings` — dependency-free hashed-n-gram embeddings
  (a legitimate technique, not a stub) so tests and offline demos can
  exercise the full embedding-driven path without network access.
"""

from src.embeddings.base import EmbeddingProvider
from src.embeddings.hashing import HashingEmbeddings
from src.embeddings.openai_embed import OpenAIEmbeddings
from src.embeddings.sentence_transformers_embed import SentenceTransformersEmbeddings
from src.embeddings.factory import get_default_embeddings

__all__ = [
    "EmbeddingProvider",
    "HashingEmbeddings",
    "OpenAIEmbeddings",
    "SentenceTransformersEmbeddings",
    "get_default_embeddings",
]
