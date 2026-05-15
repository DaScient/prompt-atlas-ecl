"""Threadsafe in-memory :class:`RunStore` — default backend."""
from __future__ import annotations

import threading
from typing import Dict, List, Optional

from src.runstore.base import RunRecord, StepRecord


class InMemoryRunStore:
    """Process-local store backed by a plain ``dict``.

    Identical observable behavior to the original ``RUNS`` global, plus a
    lock so two FastAPI workers in threaded mode don't trample each
    other's appends.
    """

    def __init__(self) -> None:
        self._runs: Dict[str, RunRecord] = {}
        self._lock = threading.Lock()

    def create(self, record: RunRecord) -> None:
        with self._lock:
            self._runs[record.run_id] = record

    def get(self, run_id: str) -> Optional[RunRecord]:
        with self._lock:
            return self._runs.get(run_id)

    def list_for_user(self, user_id: str) -> List[RunRecord]:
        with self._lock:
            return [r for r in self._runs.values() if r.user_id == user_id]

    def update_state(self, run_id: str, *, t: int, state: List[float]) -> None:
        with self._lock:
            rec = self._runs.get(run_id)
            if rec is None:
                return
            rec.t = t
            rec.state = list(state)

    def append_step(self, run_id: str, step: StepRecord) -> None:
        with self._lock:
            rec = self._runs.get(run_id)
            if rec is None:
                return
            rec.trace.append(step)

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass


__all__ = ["InMemoryRunStore"]
