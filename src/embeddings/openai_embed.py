"""OpenAI embeddings backend — lazy import."""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)


# OpenAI dims by model — kept in a small table so callers don't have to
# pass ``dim`` for the common cases. Falls back to the response length
# at runtime so unknown model names still work.
_MODEL_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddings:
    name = "openai"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        base_url: Optional[str] = None,
    ) -> None:
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._model = os.getenv("PAE_OPENAI_EMBED_MODEL", model)
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.dim = _MODEL_DIMS.get(self._model, 1536)
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise RuntimeError("OpenAIEmbeddings requires OPENAI_API_KEY")
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "openai SDK not installed; `pip install openai` or use "
                "HashingEmbeddings"
            ) from exc
        kwargs = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        client = self._ensure_client()
        # The API rejects empty strings; substitute a single space which
        # the embedder happily handles. Preserves caller batch shape.
        normalized = [t if t else " " for t in texts]
        resp = client.embeddings.create(model=self._model, input=normalized)
        vecs = [d.embedding for d in resp.data]
        if vecs:
            self.dim = len(vecs[0])
        return vecs


__all__ = ["OpenAIEmbeddings"]
