"""Phase 6 — Community Prompt Registry.

Replaces the hardcoded 4-pack stub in ``server/app.py`` with a real
registry backed by YAML/JSON files on disk. Packs are loaded once at
startup (or on first API hit) and exposed through API endpoints with
pagination + search.

Bundled packs live in ``prompts/packs/*.yaml``; operators can drop
their own packs into a directory pointed at by ``PAE_PROMPT_PACKS_DIR``
to overlay or extend the bundled set.
"""

from src.registry.models import PromptPack, PromptTemplate
from src.registry.loader import (
    PromptPackRegistry,
    get_default_pack_registry,
)

__all__ = [
    "PromptPack",
    "PromptPackRegistry",
    "PromptTemplate",
    "get_default_pack_registry",
]
