"""Phase 4 API: /runs listing, /runs/{id}/metrics, /dashboard static mount."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from server import app as app_module


@pytest.fixture
def client():
    from src.runstore import InMemoryRunStore
    from src.streaming import StreamHub

    app_module.RUN_STORE = InMemoryRunStore()
    app_module.STREAM_HUB = StreamHub()
    with TestClient(app_module.app) as c:
        yield c


def _make_run(client) -> str:
    resp = client.post(
        "/runs",
        json={"brief": {"goal": "make a dashboard"}, "config": {}},
        headers={"X-API-Key": "demo-free-key"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["run_id"]


# ----------------------------------------------------------------- /runs list


def test_runs_list_empty_when_no_runs(client):
    resp = client.get("/runs", headers={"X-API-Key": "demo-free-key"})
    assert resp.status_code == 200
    assert resp.json() == {"runs": []}


def test_runs_list_includes_owned_runs_with_summary(client):
    run_id = _make_run(client)
    client.post(f"/runs/{run_id}/step", headers={"X-API-Key": "demo-free-key"})
    client.post(f"/runs/{run_id}/step", headers={"X-API-Key": "demo-free-key"})

    resp = client.get("/runs", headers={"X-API-Key": "demo-free-key"})
    assert resp.status_code == 200
    runs = resp.json()["runs"]
    assert len(runs) == 1
    row = runs[0]
    assert row["run_id"] == run_id
    assert row["brief_goal"] == "make a dashboard"
    assert row["steps"] == 2
    assert row["latest_e_star"] is not None
    assert row["mean_e_star"] is not None


def test_runs_list_isolates_owners(client):
    _make_run(client)  # owned by u_demo
    resp = client.get("/runs", headers={"X-API-Key": "demo-pro-key"})
    assert resp.status_code == 200
    assert resp.json()["runs"] == []


def test_runs_list_requires_api_key(client):
    resp = client.get("/runs")
    assert resp.status_code == 401


# --------------------------------------------------------------- /metrics


def test_metrics_endpoint_returns_series_and_summary(client):
    run_id = _make_run(client)
    client.post(f"/runs/{run_id}/step", headers={"X-API-Key": "demo-free-key"})
    client.post(f"/runs/{run_id}/step", headers={"X-API-Key": "demo-free-key"})

    resp = client.get(f"/runs/{run_id}/metrics", headers={"X-API-Key": "demo-free-key"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["run_id"] == run_id

    # Two e_star points, one per step.
    assert len(body["e_star"]) == 2
    assert body["e_star"][0]["t"] == 1
    assert "e_star" in body["e_star"][0]

    # Two drift points; the first has no delta (no previous state).
    assert len(body["drift"]) == 2
    assert body["drift"][0]["state_delta"] is None
    assert body["drift"][1]["state_delta"] is not None

    # Summary fields are present.
    summary = body["summary"]
    assert summary["steps"] == 2
    assert summary["latest_e_star"] is not None
    assert summary["mean_e_star"] is not None


def test_metrics_rejects_other_owners(client):
    run_id = _make_run(client)
    resp = client.get(
        f"/runs/{run_id}/metrics", headers={"X-API-Key": "demo-pro-key"}
    )
    assert resp.status_code == 404


def test_metrics_404_for_missing_run(client):
    resp = client.get(
        "/runs/does-not-exist/metrics", headers={"X-API-Key": "demo-free-key"}
    )
    assert resp.status_code == 404


# --------------------------------------------------------------- /dashboard


def test_dashboard_static_files_are_mounted(client):
    """``web/`` is shipped with the repo so the mount must exist."""
    resp = client.get("/dashboard/")
    assert resp.status_code == 200
    assert b"Prompt Atlas" in resp.content


def test_dashboard_run_page_served(client):
    resp = client.get("/dashboard/run.html")
    assert resp.status_code == 200
    assert b"Run Dashboard" in resp.content
