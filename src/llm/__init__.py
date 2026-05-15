"""Phase 5 — Pluggable LLM provider abstraction.

Following the Phase 1–4 contract: optional cloud SDKs are lazy-imported and
a dependency-free fallback (``DeterministicProvider``) keeps the test suite
and offline demos working without network access or API keys.

The ``DualLLMProvider`` is the "dual-LLM" headline feature: it runs two
providers in parallel (e.g. OpenAI + Anthropic) and picks the response with
the higher heuristic confidence, exposing a single ``complete(...)`` call
to the rest of the system. Disagreement between the two providers is
recorded on the result so downstream agents (and the dashboard) can surface
the divergence.
"""

from src.llm.base import LLMProvider, LLMResponse
from src.llm.deterministic import DeterministicProvider
from src.llm.openai_provider import OpenAIProvider
from src.llm.anthropic_provider import AnthropicProvider
from src.llm.dual import DualLLMProvider
from src.llm.factory import get_default_llm

__all__ = [
    "AnthropicProvider",
    "DeterministicProvider",
    "DualLLMProvider",
    "LLMProvider",
    "LLMResponse",
    "OpenAIProvider",
    "get_default_llm",
]
