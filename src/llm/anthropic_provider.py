"""Anthropic LLM provider — lazy import, env-driven activation.

Same shape as :class:`~src.llm.openai_provider.OpenAIProvider`; kept in a
separate module so the imports are cleanly scoped and the user can omit
either SDK install without breaking the other.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from src.llm.base import LLMResponse

logger = logging.getLogger(__name__)


class AnthropicProvider:
    """Messages-API wrapper around the official ``anthropic`` SDK."""

    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-haiku-latest",
    ) -> None:
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._model = os.getenv("PAE_ANTHROPIC_MODEL", model)
        self._client = None  # lazily constructed

    # ------------------------------------------------------------------ private

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise RuntimeError(
                "AnthropicProvider requires ANTHROPIC_API_KEY to be set"
            )
        try:
            from anthropic import Anthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "anthropic SDK not installed; `pip install anthropic` or use "
                "DeterministicProvider"
            ) from exc

        self._client = Anthropic(api_key=self._api_key)
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

        try:
            message = client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system or "You are a helpful AI assistant.",
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("AnthropicProvider: completion failed: %s", exc)
            raise

        # Anthropic returns a list of content blocks; concatenate the
        # text blocks (tool-use blocks would have other ``type`` values).
        parts = []
        for block in getattr(message, "content", []) or []:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                parts.append(getattr(block, "text", "") or "")
        text = "".join(parts).strip()

        stop_reason = getattr(message, "stop_reason", "end_turn") or "end_turn"
        confidence = 0.85 if stop_reason == "end_turn" else 0.55

        usage = getattr(message, "usage", None)
        return LLMResponse(
            text=text,
            provider=self.name,
            confidence=confidence,
            meta={
                "model": self._model,
                "stop_reason": stop_reason,
                "usage": usage and {
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None),
                },
            },
        )


__all__ = ["AnthropicProvider"]
