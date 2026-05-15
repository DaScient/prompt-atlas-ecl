"""Integration tests for the FastAPI app after the Phase 3 RunStore + WS rework."""
import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from server import app as app_module


@pytest.fixture
def client():
    # Force a fresh in-memory RunStore so tests don't share state.
    from src.runstore import InMemoryRunStore
    from src.streaming import StreamHub

    app_module.RUN_STORE = InMemoryRunStore()
    app_module.STREAM_HUB = StreamHub()
    with TestClient(app_module.app) as c:
        yield c


def _make_run(client) -> str:
    resp = client.post(
        "/runs",
        json={"brief": {"goal": "test"}, "config": {}},
        headers={"X-API-Key": "demo-free-key"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["run_id"]


def test_create_run_persists_via_runstore(client):
    run_id = _make_run(client)
    rec = app_module.RUN_STORE.get(run_id)
    assert rec is not None
    assert rec.user_id == "u_demo"
    assert rec.t == 0


def test_step_appends_to_runstore_trace(client):
    run_id = _make_run(client)
    resp = client.post(f"/runs/{run_id}/step", headers={"X-API-Key": "demo-free-key"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["t"] == 1
    assert "e_star" in body

    rec = app_module.RUN_STORE.get(run_id)
    assert rec.t == 1
    assert len(rec.trace) == 1
    assert rec.trace[0].t == 1


def test_trace_endpoint_serializes_step_records(client):
    run_id = _make_run(client)
    client.post(f"/runs/{run_id}/step", headers={"X-API-Key": "demo-free-key"})
    resp = client.get(f"/runs/{run_id}/trace", headers={"X-API-Key": "demo-free-key"})
    assert resp.status_code == 200
    payload = resp.json()["run"]
    # The legacy shape is preserved: trace items are dicts with t/spec/tests/e_star.
    assert payload["t"] == 1
    assert isinstance(payload["trace"], list)
    assert payload["trace"][0]["t"] == 1
    assert "spec" in payload["trace"][0]


def test_other_users_cannot_see_each_others_runs(client):
    run_id = _make_run(client)  # owned by u_demo (free key)
    resp = client.get(
        f"/runs/{run_id}/trace",
        headers={"X-API-Key": "demo-pro-key"},  # u_demo_pro
    )
    assert resp.status_code == 404


def test_websocket_stream_pushes_new_steps(client):
    run_id = _make_run(client)

    with client.websocket_connect(
        f"/runs/{run_id}/stream?x_api_key=demo-free-key"
    ) as ws:
        snapshot = json.loads(ws.receive_text())
        assert snapshot["type"] == "snapshot"
        assert snapshot["run_id"] == run_id

        # Trigger a step via the HTTP endpoint; the WS should receive it.
        client.post(f"/runs/{run_id}/step", headers={"X-API-Key": "demo-free-key"})

        msg = json.loads(ws.receive_text())
        assert msg["type"] == "step"
        assert msg["t"] == 1
        assert "e_star" in msg


def test_websocket_rejects_missing_api_key(client):
    run_id = _make_run(client)
    with pytest.raises(Exception):
        # Starlette/TestClient surfaces the close as an exception when no
        # frames are sent before close. Either is an acceptable rejection.
        with client.websocket_connect(f"/runs/{run_id}/stream") as ws:
            ws.receive_text()


def test_websocket_rejects_wrong_owner(client):
    run_id = _make_run(client)  # u_demo
    with pytest.raises(Exception):
        with client.websocket_connect(
            f"/runs/{run_id}/stream?x_api_key=demo-pro-key"
        ) as ws:
            ws.receive_text()
