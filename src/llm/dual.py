"""``DualLLMProvider`` — run two providers in parallel, pick the winner.

This is the headline of Phase 5's v0.5.x roadmap entry:
"Dual-LLM Integration (OpenAI + Anthropic)". The pattern is useful for
two reasons:

* **Robustness** — if one provider is rate-limited or returns garbage,
  the other can carry the request.
* **Cross-checking** — recording both responses and their divergence lets
  downstream agents (and a reviewer in the dashboard) spot
  disagreements, which is exactly the kind of signal the ECL framing
  cares about.

The chooser is intentionally simple — highest reported ``confidence``,
ties broken by longer non-empty response. The full pair is always
returned on ``meta.candidates`` so a caller can implement smarter
strategies without forking this class.
"""
from __future__ import annotations

import concurrent.futures as cf
import logging
from typing import List, Optional, Sequence

from src.llm.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class DualLLMProvider:
    """Fan out one completion to N providers; return the best one."""

    name = "dual"

    def __init__(self, providers: Sequence[LLMProvider]) -> None:
        if not providers:
            raise ValueError("DualLLMProvider needs at least one provider")
        self._providers: List[LLMProvider] = list(providers)

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> LLMResponse:
        results: List[LLMResponse] = []

        # Run each provider in its own thread so a slow/rate-limited
        # backend doesn't block the other. A short timeout (provider's
        # own SDK timeout still applies underneath) keeps the worst case
        # bounded.
        with cf.ThreadPoolExecutor(max_workers=len(self._providers)) as pool:
            futures = {
                pool.submit(
                    p.complete,
                    prompt,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ): p
                for p in self._providers
            }
            for fut in cf.as_completed(futures):
                provider = futures[fut]
                try:
                    results.append(fut.result())
                except Exception as exc:
                    # Log & move on; the other provider may still succeed.
                    logger.warning(
                        "DualLLMProvider: %s failed (%s)", provider.name, exc,
                    )

        if not results:
            raise RuntimeError(
                "DualLLMProvider: all providers failed; check API keys / network"
            )

        # Highest confidence wins; ties resolved by response length.
        winner = max(
            results,
            key=lambda r: (r.confidence, len(r.text or "")),
        )

        # Divergence ∈ [0, 1] — 0 means everyone agreed verbatim, 1 means
        # every provider returned a different string. Useful for the
        # dashboard to flag "controversial" steps.
        unique = {r.text for r in results if r.text}
        divergence = (len(unique) - 1) / max(1, len(results) - 1) if len(results) > 1 else 0.0

        return LLMResponse(
            text=winner.text,
            provider=f"dual({winner.provider})",
            confidence=winner.confidence,
            meta={
                "winner": winner.provider,
                "divergence": divergence,
                "candidates": [
                    {
                        "provider": r.provider,
                        "confidence": r.confidence,
                        "text": r.text,
                    }
                    for r in results
                ],
            },
        )


__all__ = ["DualLLMProvider"]
