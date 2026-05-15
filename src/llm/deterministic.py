"""Dependency-free deterministic LLM, used as the universal fallback.

This is not a "language model" in the usual sense — it just produces a
stable, structured response derived from a hash of the prompt. That's
enough to:

* keep every Phase 5 unit test self-contained (no network, no API key);
* let demos run offline; and
* give the Orchestrator something to fan out to when neither OpenAI nor
  Anthropic credentials are present, while still exercising the full
  agent / bus / state path end-to-end.

The output is intentionally structured (looks like a tiny spec or test
plan) so downstream agents that parse JSON-ish snippets don't crash on a
random string.
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional

from src.llm.base import LLMResponse


class DeterministicProvider:
    """Stable, network-free LLM stand-in."""

    name = "deterministic"

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> LLMResponse:
        # Hash-derived "thought" so identical prompts give identical
        # output; small enough to be readable in test failures.
        digest = hashlib.sha256(((system or "") + "\n" + prompt).encode("utf-8")).hexdigest()
        tag = digest[:8]

        payload = {
            "summary": f"deterministic response {tag}",
            "key_points": [
                f"derived from prompt hash {tag}",
                f"system prefix length: {len(system) if system else 0}",
                f"prompt length: {len(prompt)}",
            ],
            "confidence_hint": (int(digest[:2], 16) % 100) / 100.0,
        }
        text = json.dumps(payload, separators=(",", ":"))
        # Clip to honour the caller's max_tokens cap (rough char ≈ token).
        text = text[: max_tokens * 4]

        return LLMResponse(
            text=text,
            provider=self.name,
            # Deliberately moderate confidence so DualLLMProvider prefers a
            # real provider's answer when one is available.
            confidence=0.3,
            meta={"hash": tag, "deterministic": True},
        )


__all__ = ["DeterministicProvider"]
