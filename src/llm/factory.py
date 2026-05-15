"""Environment-driven LLM provider selection.

``PAE_LLM_PROVIDER`` chooses the default backend:

* ``deterministic`` (default) — dependency-free, network-free.
* ``openai``                  — requires ``OPENAI_API_KEY`` + ``openai`` SDK.
* ``anthropic``               — requires ``ANTHROPIC_API_KEY`` + ``anthropic`` SDK.
* ``dual``                    — runs both real providers in parallel and
                                picks the highest-confidence response. Falls
                                back to whichever provider is reachable; if
                                neither, falls back to ``deterministic``.

The factory never raises on missing optional deps — it logs and silently
substitutes the deterministic provider — so the import-time API path
stays usable even when the operator misconfigures the env.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

from src.llm.anthropic_provider import AnthropicProvider
from src.llm.base import LLMProvider
from src.llm.deterministic import DeterministicProvider
from src.llm.dual import DualLLMProvider
from src.llm.openai_provider import OpenAIProvider

logger = logging.getLogger(__name__)


def _build_openai_if_configured() -> Optional[LLMProvider]:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        return OpenAIProvider()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("OpenAIProvider init failed: %s", exc)
        return None


def _build_anthropic_if_configured() -> Optional[LLMProvider]:
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        return AnthropicProvider()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("AnthropicProvider init failed: %s", exc)
        return None


def get_default_llm(provider_name: Optional[str] = None) -> LLMProvider:
    """Resolve the active provider, honouring env config.

    Args:
        provider_name: explicit override; falls back to
            ``PAE_LLM_PROVIDER`` env, then to ``"deterministic"``.

    Plugin providers registered under the ``llm`` namespace (see
    :mod:`src.plugins`) are tried after the built-ins; this lets users
    select a third-party model with the same ``PAE_LLM_PROVIDER`` env
    var they'd use for ``openai`` or ``anthropic``.
    """
    name = (provider_name or os.getenv("PAE_LLM_PROVIDER", "deterministic")).strip().lower()

    if name == "openai":
        return _build_openai_if_configured() or _fallback("openai missing/unset")

    if name == "anthropic":
        return _build_anthropic_if_configured() or _fallback("anthropic missing/unset")

    if name == "dual":
        backends: List[LLMProvider] = []
        oai = _build_openai_if_configured()
        ant = _build_anthropic_if_configured()
        if oai:
            backends.append(oai)
        if ant:
            backends.append(ant)
        if not backends:
            return _fallback("dual: neither provider configured")
        if len(backends) == 1:
            # Don't pay the dual-fanout cost for a single provider.
            return backends[0]
        return DualLLMProvider(backends)

    # Phase 6 — plugin providers. We look these up *before* the
    # "deterministic" default so a plugin named e.g. "ollama" can be
    # selected by setting PAE_LLM_PROVIDER=ollama.
    plugin = _try_plugin_provider(name)
    if plugin is not None:
        return plugin

    # Default — also handles "deterministic" / unknown names safely.
    if name not in {"deterministic", ""}:
        logger.info("Unknown PAE_LLM_PROVIDER=%r; using deterministic", name)
    return DeterministicProvider()


def _try_plugin_provider(name: str) -> Optional[LLMProvider]:
    """Look the name up in the plugin registry; return an instance or None."""
    if not name:
        return None
    try:
        from src.plugins import get_default_registry
    except Exception:  # pragma: no cover - circular-import defence
        return None
    rec = get_default_registry().get("llm", name)
    if rec is None:
        return None
    try:
        instance = rec.factory()
    except Exception as exc:
        logger.warning("LLM plugin %s instantiation failed: %s", name, exc)
        return None
    if not hasattr(instance, "complete"):
        logger.warning(
            "LLM plugin %s returned object without .complete(); ignoring",
            name,
        )
        return None
    return instance


def _fallback(reason: str) -> LLMProvider:
    logger.info("LLM provider falling back to deterministic: %s", reason)
    return DeterministicProvider()


__all__ = ["get_default_llm"]
