"""Tests for the Phase 2 MLflow tracker wrapper."""
from src.tracking import MLflowTracker


def test_tracker_disabled_by_default_is_noop():
    t = MLflowTracker(enabled=False)
    assert t.active is False
    # All ops must be safe to call.
    t.log_params({"a": 1, "b": "x"})
    t.log_metrics({"loss": 0.5, "garbage": "ignore"}, step=0)
    t.start_run()
    t.end_run()


def test_tracker_context_manager_is_safe_when_disabled():
    with MLflowTracker(enabled=False) as t:
        t.log_metrics({"x": 1.0}, step=0)
    # No exceptions, no state required.


def test_tracker_handles_unparseable_metric_values():
    t = MLflowTracker(enabled=False)
    # Mixed numeric / non-numeric → must not raise.
    t.log_metrics({"ok": 1, "bad": None, "alsobad": "nope"}, step=5)
