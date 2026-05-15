"""LLM provider protocol and result dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@dataclass
class LLMResponse:
    """A single completion result returned by an :class:`LLMProvider`.

    ``confidence`` is a heuristic [0, 1] score the provider attaches to its
    output — used by :class:`~src.llm.dual.DualLLMProvider` to choose
    between two parallel completions. Providers that can't introspect
    confidence should return ``0.5`` (neutral).

    ``meta`` is a free-form bag for provider-specific telemetry (model
    name, latency, token counts) that the dashboard / tracker can surface.
    """

    text: str
    provider: str
    confidence: float = 0.5
    meta: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    """Minimum contract every LLM backend must implement.

    Kept synchronous so callers don't have to thread an event loop through
    the Core stepper. Providers that wrap async SDKs should bridge with
    ``asyncio.run`` or a worker thread internally.
    """

    name: str

    def complete(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> LLMResponse: ...


__all__ = ["LLMProvider", "LLMResponse"]
