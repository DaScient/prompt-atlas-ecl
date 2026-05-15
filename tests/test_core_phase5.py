"""Phase 5 — Core.step orchestrator wiring."""
import pytest

torch = pytest.importorskip("torch")

from server.core_bridge import Core
from src.agents import Orchestrator
from src.embeddings import HashingEmbeddings
from src.llm import DeterministicProvider


def test_core_step_without_orchestrator_uses_torch_path():
    """Default behaviour (no PAE_LLM, no orchestrator) is unchanged."""
    c = Core(state_dim=32)
    out = c.step()
    assert "spec" in out and "tests" in out and "e_star" in out
    assert len(out["state"]) == 32
    # Torch path doesn't emit Phase 5 keys.
    assert "ethics" not in out


def test_core_step_with_orchestrator_and_brief_uses_orchestrator():
    orch = Orchestrator(
        llm=DeterministicProvider(),
        embeddings=HashingEmbeddings(dim=32),
    )
    c = Core(state_dim=32, orchestrator=orch)
    out = c.step(brief={"goal": "phase 5"})
    # Orchestrator path adds ethics + agents metadata.
    assert "ethics" in out
    assert "verdict" in out["ethics"]
    assert "agents" in out
    assert len(out["agents"]) == 3
    assert len(out["state"]) == 32


def test_core_step_with_orchestrator_but_no_brief_uses_torch_path():
    """Without a brief, the orchestrator has nothing to do."""
    orch = Orchestrator(
        llm=DeterministicProvider(),
        embeddings=HashingEmbeddings(dim=32),
    )
    c = Core(state_dim=32, orchestrator=orch)
    out = c.step()
    assert "ethics" not in out


def test_core_env_activates_orchestrator(monkeypatch):
    """PAE_LLM=1 wires the orchestrator automatically."""
    monkeypatch.setenv("PAE_LLM", "1")
    monkeypatch.setenv("PAE_LLM_PROVIDER", "deterministic")
    monkeypatch.setenv("PAE_EMBEDDINGS_PROVIDER", "hashing")
    c = Core(state_dim=64)
    assert c.orchestrator is not None
    out = c.step(brief={"goal": "auto"})
    assert "ethics" in out
