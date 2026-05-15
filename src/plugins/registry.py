"""Plugin registry implementation."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


# The canonical namespaces. Kept as a frozenset so typos at registration
# time fail loudly (rather than silently creating a "lmm" bucket).
_NAMESPACES = frozenset({"llm", "embeddings", "agent", "tester"})


@dataclass
class PluginRecord:
    """One registered extension."""

    namespace: str
    name: str
    factory: Callable[..., Any]
    source: str = "in-process"
    meta: Dict[str, Any] = field(default_factory=dict)


class PluginRegistry:
    """Namespaced extension registry."""

    def __init__(self) -> None:
        self._by_namespace: Dict[str, Dict[str, PluginRecord]] = {
            ns: {} for ns in _NAMESPACES
        }
        self._discovered = False

    # ----------------------------------------------------------- registration

    def register(
        self,
        namespace: str,
        name: str,
        factory: Callable[..., Any],
        *,
        source: str = "in-process",
        meta: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> PluginRecord:
        if namespace not in _NAMESPACES:
            raise ValueError(
                f"Unknown plugin namespace {namespace!r}; "
                f"valid: {sorted(_NAMESPACES)}"
            )
        if not name:
            raise ValueError("plugin name must be non-empty")
        bucket = self._by_namespace[namespace]
        if name in bucket and not overwrite:
            # Don't fail — just keep the first registration. This lets
            # an application override a built-in by registering before
            # discovery runs, without third-party plugins surprising the
            # user by silently shadowing core providers later.
            logger.info(
                "Plugin %s/%s already registered from %s; ignoring %s",
                namespace, name, bucket[name].source, source,
            )
            return bucket[name]
        rec = PluginRecord(
            namespace=namespace,
            name=name,
            factory=factory,
            source=source,
            meta=dict(meta or {}),
        )
        bucket[name] = rec
        return rec

    # -------------------------------------------------------------- lookup

    def get(self, namespace: str, name: str) -> Optional[PluginRecord]:
        if namespace not in _NAMESPACES:
            return None
        return self._by_namespace[namespace].get(name)

    def list(self, namespace: Optional[str] = None) -> List[PluginRecord]:
        if namespace is not None:
            if namespace not in _NAMESPACES:
                return []
            return list(self._by_namespace[namespace].values())
        out: List[PluginRecord] = []
        for bucket in self._by_namespace.values():
            out.extend(bucket.values())
        return out

    def namespaces(self) -> List[str]:
        return sorted(_NAMESPACES)

    # --------------------------------------------------------- entry points

    def discover_entry_points(
        self,
        *,
        group_prefix: str = "prompt_atlas",
        force: bool = False,
    ) -> int:
        """Walk ``importlib.metadata`` entry points and register them.

        Returns the number of plugins newly registered. Safe to call
        multiple times — the second call is a no-op unless
        ``force=True``.

        Entry point groups follow the convention
        ``prompt_atlas.<namespace>``, e.g. ``prompt_atlas.llm`` for
        custom LLM providers. The entry's loaded object is treated as
        the factory.
        """
        if self._discovered and not force:
            return 0

        # Lazy import — Python <3.10 needs importlib_metadata, which we
        # don't want as a hard dep.
        try:
            from importlib import metadata as importlib_metadata  # type: ignore
        except Exception:  # pragma: no cover
            logger.info("importlib.metadata unavailable; skipping plugin discovery")
            self._discovered = True
            return 0

        count = 0
        for ns in _NAMESPACES:
            group = f"{group_prefix}.{ns}"
            try:
                eps = importlib_metadata.entry_points(group=group)  # type: ignore[arg-type]
            except TypeError:  # pragma: no cover - py<3.10
                eps = importlib_metadata.entry_points().get(group, [])  # type: ignore
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("entry_points(%s) failed: %s", group, exc)
                continue

            for ep in eps:
                try:
                    factory = ep.load()
                except Exception as exc:
                    # Bad plugin must NEVER break us.
                    logger.warning(
                        "plugin %s in group %s failed to load: %s",
                        getattr(ep, "name", "?"), group, exc,
                    )
                    continue
                try:
                    self.register(
                        ns,
                        ep.name,
                        factory,
                        source=f"entry_point:{group}",
                    )
                    count += 1
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("plugin %s/%s register failed: %s", ns, ep.name, exc)

        self._discovered = True
        return count


# --------------------------------------------------------------------- module API

_GLOBAL: Optional[PluginRegistry] = None


def get_default_registry() -> PluginRegistry:
    """Return the process-wide plugin registry, creating it on first use.

    On first creation, entry-point discovery runs *only* when
    ``PAE_PLUGINS=1`` is set. Discovery is opt-in because importing
    arbitrary third-party code at server startup is a security and
    stability concern; the default deployment runs only built-ins.
    """
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = PluginRegistry()
        if os.getenv("PAE_PLUGINS", "0") == "1":
            try:
                _GLOBAL.discover_entry_points()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("plugin discovery failed: %s", exc)
    return _GLOBAL


def register(
    namespace: str,
    name: str,
    *,
    overwrite: bool = False,
    meta: Optional[Dict[str, Any]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator form: ``@register("llm", "my-llm")``."""

    def _decorate(factory: Callable[..., Any]) -> Callable[..., Any]:
        get_default_registry().register(
            namespace, name, factory,
            source="decorator", meta=meta, overwrite=overwrite,
        )
        return factory

    return _decorate


def discover_entry_points(*, force: bool = False) -> int:
    return get_default_registry().discover_entry_points(force=force)


__all__ = [
    "PluginRecord",
    "PluginRegistry",
    "discover_entry_points",
    "get_default_registry",
    "register",
]
