"""Phase 6 — API tests for the new prompt-pack + plugin endpoints."""
import pytest

httpx = pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from server.app import app
from src.plugins import get_default_registry


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_list_prompt_packs_returns_bundled(client):
    r = client.get("/prompt-packs")
    assert r.status_code == 200
    body = r.json()
    ids = {p["id"] for p in body["packs"]}
    # All four legacy IDs are present (back-compat).
    assert {"myth-1", "science-1", "psych-1", "purpose-1"}.issubset(ids)
    # Pagination metadata exists.
    assert "total" in body and "offset" in body and "limit" in body


def test_list_prompt_packs_search(client):
    r = client.get("/prompt-packs", params={"q": "empathy"})
    assert r.status_code == 200
    ids = {p["id"] for p in r.json()["packs"]}
    assert "psych-1" in ids


def test_list_prompt_packs_filter_domain(client):
    r = client.get("/prompt-packs", params={"domain": "science"})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 1
    for p in body["packs"]:
        assert p["domain"] == "science"


def test_list_prompt_packs_pagination(client):
    r = client.get("/prompt-packs", params={"limit": 1, "offset": 0})
    assert r.status_code == 200
    body = r.json()
    assert len(body["packs"]) == 1
    assert body["total"] >= 4


def test_get_prompt_pack_detail(client):
    r = client.get("/prompt-packs/myth-1")
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == "myth-1"
    assert isinstance(body["prompts"], list)
    assert len(body["prompts"]) >= 1


def test_get_prompt_pack_not_found(client):
    r = client.get("/prompt-packs/not-a-real-pack")
    assert r.status_code == 404


def test_get_prompt_template(client):
    r = client.get("/prompt-packs/myth-1/prompts/archetype_audit")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "archetype_audit"
    assert "body" in body


def test_get_prompt_template_not_found(client):
    r = client.get("/prompt-packs/myth-1/prompts/does-not-exist")
    assert r.status_code == 404


def test_list_plugins_endpoint(client):
    # Add a plugin so we can see it surface.
    get_default_registry().register(
        "llm", "_phase6_api_test_plugin",
        lambda: None,
        overwrite=True,
        meta={"hello": "world"},
    )
    r = client.get("/plugins")
    assert r.status_code == 200
    body = r.json()
    assert "namespaces" in body
    assert {"llm", "embeddings", "agent", "tester"} == set(body["namespaces"])
    names = {(p["namespace"], p["name"]) for p in body["plugins"]}
    assert ("llm", "_phase6_api_test_plugin") in names
    # The factory must NOT be in the response.
    for p in body["plugins"]:
        assert "factory" not in p
