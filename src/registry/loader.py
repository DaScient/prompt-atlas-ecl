"""Disk-backed prompt-pack loader.

Reads ``*.yaml`` / ``*.yml`` / ``*.json`` files from one or more
directories and exposes them as :class:`PromptPack` instances. The
bundled directory (``prompts/packs/`` under the repo root) is always
loaded; additional directories from ``PAE_PROMPT_PACKS_DIR`` (colon-
separated) overlay on top — later entries override earlier ones,
matching the standard Unix path convention.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.registry.models import PromptPack

logger = logging.getLogger(__name__)


# Resolve the bundled pack dir relative to this file so it works
# regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUNDLED_DIR = _REPO_ROOT / "prompts" / "packs"


class PromptPackRegistry:
    """In-memory index of all known prompt packs."""

    def __init__(self, dirs: Sequence[Path] = ()) -> None:
        self._dirs: List[Path] = [Path(d) for d in dirs]
        self._packs: Dict[str, PromptPack] = {}
        self._loaded = False

    # ----------------------------------------------------- loading

    def load(self, *, force: bool = False) -> int:
        """Walk all configured dirs and load every pack file.

        Returns the number of packs in the registry after load. Idempotent
        unless ``force=True``.
        """
        if self._loaded and not force:
            return len(self._packs)
        if force:
            self._packs.clear()

        for d in self._dirs:
            if not d.is_dir():
                logger.info("prompt-packs dir not found, skipping: %s", d)
                continue
            for path in sorted(d.iterdir()):
                if path.is_file() and path.suffix.lower() in {".yaml", ".yml", ".json"}:
                    pack = _read_pack_file(path)
                    if pack is None:
                        continue
                    if pack.id in self._packs:
                        logger.info(
                            "prompt-pack %s overridden by %s", pack.id, path,
                        )
                    self._packs[pack.id] = pack

        self._loaded = True
        return len(self._packs)

    # ----------------------------------------------------- lookup

    def get(self, pack_id: str) -> Optional[PromptPack]:
        self.load()
        return self._packs.get(pack_id)

    def list(
        self,
        *,
        domain: Optional[str] = None,
        tag: Optional[str] = None,
        query: Optional[str] = None,
    ) -> List[PromptPack]:
        """Return packs matching optional filters.

        ``query`` matches case-insensitively against id, title, and
        description. Filters are combined with AND.
        """
        self.load()
        out: List[PromptPack] = []
        q = (query or "").strip().lower()
        for pack in self._packs.values():
            if domain and pack.domain != domain:
                continue
            if tag and tag not in pack.tags:
                continue
            if q:
                hay = " ".join([
                    pack.id, pack.title, pack.description, " ".join(pack.tags),
                ]).lower()
                if q not in hay:
                    continue
            out.append(pack)
        # Stable ordering — id is unique and url-safe, so sort by it.
        out.sort(key=lambda p: p.id)
        return out

    def all_ids(self) -> List[str]:
        self.load()
        return sorted(self._packs.keys())


# ------------------------------------------------------------- helpers


def _read_pack_file(path: Path) -> Optional[PromptPack]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("cannot read %s: %s", path, exc)
        return None

    suffix = path.suffix.lower()
    try:
        if suffix in {".yaml", ".yml"}:
            data = _yaml_load(text)
        else:
            data = json.loads(text)
    except Exception as exc:
        logger.warning("cannot parse %s: %s", path, exc)
        return None

    if not isinstance(data, dict):
        logger.warning("pack file %s did not decode to a dict; skipped", path)
        return None

    try:
        return PromptPack.model_validate(data)
    except Exception as exc:
        logger.warning("pack file %s failed validation: %s", path, exc)
        return None


def _yaml_load(text: str) -> Any:
    """Load YAML if PyYAML is installed; otherwise try JSON.

    The repo already depends on PyYAML elsewhere, but the loader stays
    defensive so a stripped-down install (e.g. JSON-only packs) still
    works.
    """
    try:
        import yaml  # type: ignore
    except ImportError:  # pragma: no cover
        return json.loads(text)
    return yaml.safe_load(text)


# ------------------------------------------------------------- module API


_GLOBAL: Optional[PromptPackRegistry] = None


def _resolve_dirs() -> List[Path]:
    dirs: List[Path] = [_BUNDLED_DIR]
    overlay = os.getenv("PAE_PROMPT_PACKS_DIR", "")
    if overlay:
        # Colon-separated on Unix, semicolon on Windows — split on both.
        for entry in overlay.replace(";", ":").split(":"):
            entry = entry.strip()
            if entry:
                dirs.append(Path(entry))
    return dirs


def get_default_pack_registry() -> PromptPackRegistry:
    """Process-wide pack registry; constructed lazily."""
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = PromptPackRegistry(_resolve_dirs())
    return _GLOBAL


__all__ = ["PromptPackRegistry", "get_default_pack_registry"]
