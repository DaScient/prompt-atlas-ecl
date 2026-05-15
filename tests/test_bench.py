"""Phase 6 — benchmark harness tests."""
import json

import pytest

torch = pytest.importorskip("torch")

from scripts.bench import _drift_series, run_bench


def test_drift_series_basic():
    assert _drift_series([]) == []
    assert _drift_series([[0.0, 0.0]]) == []
    drift = _drift_series([[0.0, 0.0], [3.0, 4.0]])
    assert drift == pytest.approx([5.0])  # 3-4-5 triangle


def test_drift_series_handles_uneven_lengths():
    # Different-length states shouldn't crash; the shorter length wins.
    drift = _drift_series([[1.0, 2.0, 3.0], [1.0, 2.0]])
    assert drift == pytest.approx([0.0])


def test_run_bench_torch_path_is_reproducible_shape():
    out = run_bench(steps=4, brief={"goal": "x"}, state_dim=16, use_llm=False)
    assert out["version"] == 1
    assert out["config"]["steps"] == 4
    assert out["config"]["state_dim"] == 16
    assert out["config"]["use_llm"] is False
    assert len(out["metrics"]["e_star_series"]) == 4
    # drift_series has steps-1 entries.
    assert len(out["metrics"]["drift_series"]) == 3
    assert out["metrics"]["wallclock_seconds"] >= 0.0
    assert out["metrics"]["steps_per_second"] >= 0.0


def test_run_bench_orchestrator_path(monkeypatch):
    monkeypatch.setenv("PAE_LLM_PROVIDER", "deterministic")
    monkeypatch.setenv("PAE_EMBEDDINGS_PROVIDER", "hashing")
    out = run_bench(
        steps=2,
        brief={"goal": "phase 6 bench"},
        state_dim=32,
        use_llm=True,
    )
    assert out["config"]["use_llm"] is True
    assert len(out["metrics"]["e_star_series"]) == 2


def test_run_bench_output_is_json_serializable():
    out = run_bench(steps=2, brief={"goal": "x"}, state_dim=8, use_llm=False)
    # Must round-trip through JSON cleanly (matches schema contract).
    text = json.dumps(out)
    again = json.loads(text)
    assert again["version"] == 1
