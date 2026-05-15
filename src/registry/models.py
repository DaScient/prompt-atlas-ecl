"""Pydantic models for the prompt-pack registry."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class PromptTemplate(BaseModel):
    """A single reusable prompt template inside a pack.

    ``body`` is plain text that may contain ``{placeholder}`` slots
    rendered by the caller via :meth:`PromptPack.render`. Slots are
    declared explicitly on ``inputs`` for self-documentation and so the
    UI can surface a form.
    """

    name: str
    body: str
    description: str = ""
    inputs: List[str] = Field(default_factory=list)
    role: str = "user"  # "user" / "system" — advisory


class PromptPack(BaseModel):
    """A versioned, addressable bundle of related prompt templates."""

    id: str
    version: str = "0.1.0"
    title: str
    domain: str = "general"
    tags: List[str] = Field(default_factory=list)
    description: str = ""
    author: str = "community"
    license: str = "MIT"
    prompts: List[PromptTemplate] = Field(default_factory=list)
    defaults: Dict[str, Any] = Field(default_factory=dict)

    # -------- light validation --------

    @field_validator("id")
    @classmethod
    def _id_is_slug(cls, v: str) -> str:
        if not v:
            raise ValueError("pack id must be non-empty")
        # Keep IDs URL-safe so they can be path segments.
        for ch in v:
            if not (ch.isalnum() or ch in "-_."):
                raise ValueError(
                    f"pack id {v!r} contains illegal char {ch!r}; "
                    "use [A-Za-z0-9._-]"
                )
        return v

    # -------- helpers --------

    def get_prompt(self, name: str) -> Optional[PromptTemplate]:
        for p in self.prompts:
            if p.name == name:
                return p
        return None

    def render(self, prompt_name: str, **kwargs: Any) -> str:
        """Render a prompt by name with ``str.format_map`` substitution.

        Missing slots fall back to the pack's ``defaults`` so callers
        don't have to repeat boilerplate. Unknown slots raise
        ``KeyError`` so typos in the caller don't silently produce a
        broken prompt.
        """
        tmpl = self.get_prompt(prompt_name)
        if tmpl is None:
            raise KeyError(f"prompt {prompt_name!r} not in pack {self.id!r}")
        # Defaults supply fallback values; caller args win.
        values: Dict[str, Any] = {**self.defaults, **kwargs}
        return tmpl.body.format_map(_SafeFormatDict(values))


class _SafeFormatDict(dict):
    """Dict that raises a helpful error on missing keys."""

    def __missing__(self, key):
        raise KeyError(f"prompt template needs value for {{{key}}}")


__all__ = ["PromptPack", "PromptTemplate"]
