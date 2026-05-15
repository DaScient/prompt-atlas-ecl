"""OpenAI LLM provider — lazy import, env-driven activation.

``OpenAIProvider`` is constructed eagerly but the ``openai`` SDK is
imported on first ``complete(...)`` call. This mirrors the Phase 1
pattern used for Qdrant and NATS: optional dependencies don't crash an
import-time path even when they're not installed.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from src.llm.base import LLMResponse

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """Chat-completions wrapper around the official ``openai`` SDK."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
    ) -> None:
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._model = os.getenv("PAE_OPENAI_MODEL", model)
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._client = None  # lazily constructed

    # ------------------------------------------------------------------ private

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise RuntimeError(
                "OpenAIProvider requires OPENAI_API_KEY to be set"
            )
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised in fallback path
            raise RuntimeError(
                "openai SDK not installed; `pip install openai` or use "
                "DeterministicProvider"
            ) from exc

        kwargs = {"api_key": self._api_key}
        if self._base_url:
            kwargs["base_url"] = self._base_url
        self._client = OpenAI(**kwargs)
        return self._client

    # ------------------------------------------------------------------- public

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> LLMResponse:
        client = self._ensure_client()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            completion = client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("OpenAIProvider: completion failed: %s", exc)
            raise

        choice = completion.choices[0]
        text = (choice.message.content or "").strip()

        # finish_reason "stop" is a strong signal the model finished
        # cleanly. "length" means we got cut off — lower confidence.
        finish = getattr(choice, "finish_reason", "stop") or "stop"
        confidence = 0.85 if finish == "stop" else 0.55

        return LLMResponse(
            text=text,
            provider=self.name,
            confidence=confidence,
            meta={
                "model": self._model,
                "finish_reason": finish,
                "usage": getattr(completion, "usage", None) and {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                },
            },
        )


__all__ = ["OpenAIProvider"]
