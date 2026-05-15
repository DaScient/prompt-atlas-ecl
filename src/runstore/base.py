"""RunStore protocol and shared dataclasses.

The protocol intentionally exposes a minimal surface — just what
``server/app.py`` needs — so swapping backends remains cheap.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class StepRecord:
    """A single ECL step appended to a run's trace."""

    t: int
    spec: Dict[str, Any]
    tests: List[Dict[str, Any]]
    e_star: float


@dataclass
class RunRecord:
    """The full server-side state for a single run.

    Mirrors the keys the old in-process ``RUNS[run_id]`` dict used to
    carry, with an explicit dataclass so backends can serialize/deserialize
    consistently.
    """

    run_id: str
    user_id: str
    plan: str
    brief: Dict[str, Any]
    prompt_pack_id: Optional[str]
    config: Dict[str, Any]
    t: int = 0
    state: List[float] = field(default_factory=list)
    trace: List[StepRecord] = field(default_factory=list)


@runtime_checkable
class RunStore(Protocol):
    """Minimal CRUD contract for run persistence."""

    def create(self, record: RunRecord) -> None: ...

    def get(self, run_id: str) -> Optional[RunRecord]: ...

    def update_state(self, run_id: str, *, t: int, state: List[float]) -> None: ...

    def append_step(self, run_id: str, step: StepRecord) -> None: ...

    def close(self) -> None: ...


__all__ = ["RunRecord", "RunStore", "StepRecord"]
