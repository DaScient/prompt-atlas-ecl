"""Phase 3 — pluggable persistence for ECL runs.

The API used to keep all run state in a process-global ``dict``. That works
for local demos but loses everything on restart and can't be shared across
worker processes. Phase 3 introduces a small :class:`RunStore` protocol with
two implementations:

* :class:`InMemoryRunStore` — the default, behaviorally identical to the
  old dict-based store.
* :class:`SQLRunStore` — SQLAlchemy-backed, activates automatically when
  ``PAE_DATABASE_URL`` is set (sqlite, Postgres, etc.). Falls back to the
  in-memory store at runtime if SQLAlchemy isn't installed or the URL is
  unreachable, mirroring the optional-dep contract used by Phase 1's
  NATS / Qdrant integrations.

All public access goes through :func:`get_default_runstore`.
"""

from src.runstore.base import RunRecord, RunStore, StepRecord
from src.runstore.memory import InMemoryRunStore
from src.runstore.sql_store import SQLRunStore
from src.runstore.factory import get_default_runstore

__all__ = [
    "RunRecord",
    "RunStore",
    "StepRecord",
    "InMemoryRunStore",
    "SQLRunStore",
    "get_default_runstore",
]
