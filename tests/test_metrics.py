"""Tests for the Phase 4 metric derivations."""
import math

from src.metrics import (
    e_star_series,
    latent_drift_series,
    summarize_run,
)
from src.runstore import RunRecord, StepRecord


def _record_with_states(states, e_stars=None):
    """Build a RunRecord whose trace carries per-step state vectors."""
    e_stars = e_stars if e_stars is not None else [0.5] * len(states)
    return RunRecord(
        run_id="r",
        user_id="u",
        plan="free",
        brief={"goal": "test"},
        prompt_pack_id=None,
        config={},
        t=len(states),
        state=list(states[-1]) if states else [],
        trace=[
            StepRecord(t=i + 1, spec={}, tests=[], e_star=e_stars[i], state=list(states[i]))
            for i in range(len(states))
        ],
    )


# --------------------------------------------------------------- e_star_series


def test_e_star_series_projects_each_step():
    rec = _record_with_states([[0.0], [0.0], [0.0]], e_stars=[0.1, 0.2, 0.35])
    pts = e_star_series(rec)
    assert [p.t for p in pts] == [1, 2, 3]
    assert [p.e_star for p in pts] == [0.1, 0.2, 0.35]


def test_e_star_series_empty_trace_returns_empty():
    rec = RunRecord(
        run_id="r", user_id="u", plan="free",
        brief={}, prompt_pack_id=None, config={}, t=0, state=[], trace=[],
    )
    assert e_star_series(rec) == []


# ----------------------------------------------------------- latent_drift_series


def test_latent_drift_first_step_has_no_delta():
    rec = _record_with_states([[3.0, 4.0]])  # norm = 5
    pts = latent_drift_series(rec)
    assert len(pts) == 1
    assert pts[0].t == 1
    assert math.isclose(pts[0].state_norm, 5.0)
    assert pts[0].state_delta is None


def test_latent_drift_subsequent_steps_compute_delta():
    rec = _record_with_states([[3.0, 4.0], [6.0, 8.0]])
    pts = latent_drift_series(rec)
    assert math.isclose(pts[0].state_norm, 5.0)
    assert math.isclose(pts[1].state_norm, 10.0)
    # ‖(6-3, 8-4)‖ = ‖(3,4)‖ = 5
    assert math.isclose(pts[1].state_delta, 5.0)


def test_latent_drift_legacy_steps_without_state_use_final_state_on_last_step():
    """Pre-Phase-4 step records have state=None; the final point should
    still report ``state_norm`` from the run-level ``state`` field."""
    rec = RunRecord(
        run_id="r", user_id="u", plan="free",
        brief={}, prompt_pack_id=None, config={},
        t=2, state=[3.0, 4.0],
        trace=[
            StepRecord(t=1, spec={}, tests=[], e_star=0.1, state=None),
            StepRecord(t=2, spec={}, tests=[], e_star=0.2, state=None),
        ],
    )
    pts = latent_drift_series(rec)
    assert len(pts) == 2
    assert pts[0].state_norm == 0.0
    assert pts[0].state_delta is None
    # The final step inherits the run-level state.
    assert math.isclose(pts[1].state_norm, 5.0)


def test_latent_drift_tolerates_dimension_mismatch():
    rec = _record_with_states([[1.0, 2.0, 3.0], [1.0, 2.0]])
    pts = latent_drift_series(rec)
    # ‖(0,0)‖ over the overlapping prefix
    assert math.isclose(pts[1].state_delta, 0.0)


# -------------------------------------------------------------- summarize_run


def test_summarize_run_aggregates_basic_stats():
    rec = _record_with_states(
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        e_stars=[0.2, 0.4, 0.6],
    )
    s = summarize_run(rec)
    assert s.steps == 3
    assert math.isclose(s.latest_e_star, 0.6)
    assert math.isclose(s.mean_e_star, (0.2 + 0.4 + 0.6) / 3)
    assert math.isclose(s.final_state_norm, 3.0)
    # Two deltas: 1.0 and 1.0
    assert math.isclose(s.mean_state_delta, 1.0)


def test_summarize_empty_run_is_safe():
    rec = RunRecord(
        run_id="r", user_id="u", plan="free",
        brief={}, prompt_pack_id=None, config={}, t=0, state=[], trace=[],
    )
    s = summarize_run(rec)
    assert s.steps == 0
    assert s.latest_e_star is None
    assert s.mean_e_star is None
    assert s.final_state_norm is None
    assert s.mean_state_delta is None
