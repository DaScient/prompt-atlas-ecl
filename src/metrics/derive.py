"""Implementation of the Phase 4 metric derivations.

Kept as a stand-alone module (not collapsed into ``__init__.py``) so unit
tests can import the helpers without dragging in the full re-export
surface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

from src.runstore.base import RunRecord, StepRecord


# --------------------------------------------------------------------- types


@dataclass
class EStarPoint:
    """A single point on the E-Star line chart."""

    t: int
    e_star: float


@dataclass
class LatentDriftPoint:
    """A single point on the latent drift chart.

    ``state_norm`` is the L2 magnitude of the latent state at step ``t``.
    ``state_delta`` is the L2 distance ``‖state_t − state_{t-1}‖`` —
    i.e. how much the state moved between consecutive ECL ticks. At ``t = 0``
    (or whenever a previous state isn't available) ``state_delta`` is
    ``None`` so the chart can render a gap rather than a misleading 0.
    """

    t: int
    state_norm: float
    state_delta: Optional[float]


@dataclass
class RunSummary:
    """High-level metrics for a run, suitable for an at-a-glance card."""

    run_id: str
    steps: int
    latest_e_star: Optional[float]
    mean_e_star: Optional[float]
    final_state_norm: Optional[float]
    mean_state_delta: Optional[float]


# ---------------------------------------------------------------- primitives


def _l2(vec: Sequence[float]) -> float:
    return math.sqrt(sum(float(v) * float(v) for v in vec))


def _l2_delta(a: Sequence[float], b: Sequence[float]) -> float:
    # Tolerate dimension mismatch by truncating to the shorter vector
    # rather than raising — keeps the chart usable across schema changes.
    n = min(len(a), len(b))
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(n)))


# --------------------------------------------------------------------- series


def e_star_series(record: RunRecord) -> List[EStarPoint]:
    """Project the trace down to ``[{t, e_star}, ...]`` for charting."""
    return [EStarPoint(t=st.t, e_star=float(st.e_star)) for st in record.trace]


def latent_drift_series(record: RunRecord) -> List[LatentDriftPoint]:
    """Compute latent-state magnitude and drift across the trace.

    The :class:`StepRecord` schema doesn't carry the per-step state
    vector (only the *latest* state is on :class:`RunRecord`), so for
    the historical trace we report:

    * ``state_norm`` of the latest persisted state on the *final* step.
    * ``state_norm = 0`` and ``state_delta = None`` on every earlier step.

    Once :class:`StepRecord` gains an optional ``state`` field, this
    function will compute true per-step drift; the dataclass shape stays
    the same. The dashboard can render either case unchanged.
    """
    if not record.trace:
        return []

    points: List[LatentDriftPoint] = []
    prev_state: Optional[Sequence[float]] = None

    for idx, st in enumerate(record.trace):
        # ``StepRecord.state`` is the canonical attribute (added in
        # Phase 4) but ``getattr`` is used here so this function still
        # works if a caller hands us a duck-typed step object from an
        # older serialization that doesn't carry the field at all.
        state_vec = getattr(st, "state", None)
        if state_vec is None and idx == len(record.trace) - 1:
            # Fall back to the latest persisted state for the final step.
            state_vec = record.state

        if state_vec is None:
            points.append(LatentDriftPoint(t=st.t, state_norm=0.0, state_delta=None))
            prev_state = None
            continue

        norm = _l2(state_vec)
        delta = _l2_delta(prev_state, state_vec) if prev_state is not None else None
        points.append(LatentDriftPoint(t=st.t, state_norm=norm, state_delta=delta))
        prev_state = state_vec

    return points


def summarize_run(record: RunRecord) -> RunSummary:
    """Compute an at-a-glance summary of a run."""
    trace = record.trace
    if not trace:
        return RunSummary(
            run_id=record.run_id,
            steps=0,
            latest_e_star=None,
            mean_e_star=None,
            final_state_norm=_l2(record.state) if record.state else None,
            mean_state_delta=None,
        )

    e_stars = [float(st.e_star) for st in trace]
    drift_pts = latent_drift_series(record)
    deltas = [p.state_delta for p in drift_pts if p.state_delta is not None]

    return RunSummary(
        run_id=record.run_id,
        steps=len(trace),
        latest_e_star=e_stars[-1],
        mean_e_star=sum(e_stars) / len(e_stars),
        final_state_norm=_l2(record.state) if record.state else None,
        mean_state_delta=(sum(deltas) / len(deltas)) if deltas else None,
    )


__all__ = [
    "EStarPoint",
    "LatentDriftPoint",
    "RunSummary",
    "e_star_series",
    "latent_drift_series",
    "summarize_run",
]
