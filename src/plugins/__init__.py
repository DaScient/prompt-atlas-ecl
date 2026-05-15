"""Phase 6 — pluggable extension registry.

The Phase 6 roadmap item "Plugin Ecosystem" needs a way for third-party
packages to register their own LLM providers, embedding backends, agents,
or testers without forking the repo. The :class:`PluginRegistry` here
gives us that:

* **In-process registration** — callers (or this package's own modules)
  call :func:`register` to add an extension under a namespace.
* **Entry-point discovery** — :func:`discover_entry_points` walks
  ``importlib.metadata`` entry points under ``prompt_atlas.<namespace>``
  and registers anything it finds. Failed plugin imports are *logged
  and skipped* — never raised — so a broken third-party install can't
  take down the API.

The registry is consulted by :func:`src.llm.factory.get_default_llm` and
:func:`src.embeddings.factory.get_default_embeddings` so plugin
providers can be selected via the same env-var path as built-ins.
"""

from src.plugins.registry import (
    PluginRegistry,
    PluginRecord,
    get_default_registry,
    register,
    discover_entry_points,
)

__all__ = [
    "PluginRecord",
    "PluginRegistry",
    "discover_entry_points",
    "get_default_registry",
    "register",
]
