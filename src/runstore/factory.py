"""Backend selection for :mod:`src.runstore`.

The factory honors ``PAE_DATABASE_URL`` to decide whether to spin up the
SQL backend, with a graceful in-memory fallback if SQLAlchemy isn't
available or the URL fails to open.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from src.runstore.base import RunStore
from src.runstore.memory import InMemoryRunStore

logger = logging.getLogger(__name__)


def get_default_runstore(database_url: Optional[str] = None) -> RunStore:
    """Return the configured :class:`RunStore`.

    Resolution order:

    1. Explicit ``database_url`` argument.
    2. ``PAE_DATABASE_URL`` environment variable.
    3. In-memory fallback.
    """
    url = database_url if database_url is not None else os.getenv("PAE_DATABASE_URL")
    if not url:
        return InMemoryRunStore()

    # Lazy import keeps the optional SQLAlchemy dep truly optional.
    try:
        from src.runstore.sql_store import SQLRunStore

        return SQLRunStore(url)
    except ImportError:
        logger.warning(
            "PAE_DATABASE_URL is set but SQLAlchemy is not installed; "
            "falling back to InMemoryRunStore. `pip install sqlalchemy` to enable persistence."
        )
        return InMemoryRunStore()
    except Exception as exc:  # pragma: no cover - depends on driver/env
        logger.warning(
            "Failed to initialize SQLRunStore (%s); falling back to InMemoryRunStore.",
            exc,
        )
        return InMemoryRunStore()


__all__ = ["get_default_runstore"]
