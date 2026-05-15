"""Phase 5 — agents + orchestrator tests."""
import json

import pytest

from src.agents import Orchestrator
from src.agents.orchestrator import (
    EthicsAgent,
    TesterAgent,
    WriterAgent,
    _parse_json,
)
from src.embeddings import HashingEmbeddings
from src.llm import DeterministicProvider, LLMResponse


# ----------------------------------------------------- _parse_json helper


def test_parse_json_handles_pure_json():
    assert _parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_handles_fenced_code_block():
    s = "Here you go:\n```json\n{\"a\": 1}\n```\nThanks."
    assert _parse_json(s) == {"a": 1}


def test_parse_json_handles_bare_json_in_prose():
    s = "Sure! {\"a\": 1, \"b\": [2, 3]} done."
    assert _parse_json(s) == {"a": 1, "b": [2, 3]}


def test_parse_json_returns_none_for_unparseable():
    assert _parse_json("just a sentence") is None
    assert _parse_json("") is None


# ----------------------------------------------------- LLM doubles


class _ScriptedLLM:
    """LLM that returns a queued response per call (or repeats the last)."""

    def __init__(self, responses):
        self.name = "scripted"
        self._responses = list(responses)
        self.calls = []

    def complete(self, prompt, *, system=None, max_tokens=512, temperature=0.2):
        self.calls.append({"system": system, "prompt": prompt})
        text = self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]
        return LLMResponse(text=text, provider=self.name, confidence=0.8)


# ----------------------------------------------------- individual agents


def test_writer_uses_llm_json_when_valid():
    llm = _ScriptedLLM([
        json.dumps({
            "assumptions": ["x"],
            "data": {"sources": ["custom"]},
            "steps": ["s1"],
            "interfaces": ["api"],
            "acceptance": ["ok"],
            "risks": ["r"],
        })
    ])
    w = WriterAgent(llm)
    res = w.draft({"goal": "test"})
    assert res.role == "writer"
    assert res.payload["data"]["sources"] == ["custom"]
    assert res.payload["assumptions"] == ["x"]
    # System prompt was sent
    assert "Writer" in (llm.calls[0]["system"] or "")


def test_writer_falls_back_when_llm_returns_garbage():
    w = WriterAgent(_ScriptedLLM(["I'm not JSON at all"]))
    res = w.draft({"goal": "build a thing"})
    # Falls back to the default scaffold but the goal is reflected.
    assert "build a thing" in res.payload["assumptions"][0]
    assert "acceptance" in res.payload
    assert "risks" in res.payload


def test_writer_fills_missing_keys_from_partial_response():
    llm = _ScriptedLLM([json.dumps({"assumptions": ["just one key"]})])
    w = WriterAgent(llm)
    res = w.draft({"goal": "g"})
    for key in ("data", "steps", "interfaces", "acceptance", "risks"):
        assert key in res.payload


def test_tester_returns_normalized_list():
    llm = _ScriptedLLM([
        json.dumps([
            {"name": "t1", "checks": ["c1"]},
            {"checks": ["c2"]},  # missing name → coerced
        ])
    ])
    t = TesterAgent(llm)
    res = t.draft({"acceptance": ["x"]})
    tests = res.payload["tests"]
    assert len(tests) == 2
    assert tests[0]["name"] == "t1"
    assert tests[1]["name"].startswith("test_")


def test_tester_falls_back_to_default_tests():
    t = TesterAgent(_ScriptedLLM(["not json"]))
    res = t.draft({"risks": ["r"]})
    assert len(res.payload["tests"]) >= 1


def test_ethics_normalises_verdict():
    llm = _ScriptedLLM([
        json.dumps({"concerns": ["c"], "mitigations": ["m"], "verdict": "APPROVE"})
    ])
    e = EthicsAgent(llm)
    res = e.review({})
    assert res.payload["verdict"] == "approve"


def test_ethics_rejects_invalid_verdict():
    llm = _ScriptedLLM([
        json.dumps({"verdict": "obliterate"})
    ])
    e = EthicsAgent(llm)
    res = e.review({})
    assert res.payload["verdict"] == "approve"


# ----------------------------------------------------- orchestrator


def test_orchestrator_runs_end_to_end_with_deterministic_llm():
    orch = Orchestrator(
        llm=DeterministicProvider(),
        embeddings=HashingEmbeddings(dim=64),
    )
    out = orch.run_step({"goal": "build phase 5"})
    assert isinstance(out.spec, dict)
    assert isinstance(out.tests, list)
    assert isinstance(out.ethics, dict)
    assert 0.0 <= out.e_star <= 2.0
    assert len(out.state) == 64
    assert len(out.agents) == 3
    assert [a.role for a in out.agents] == ["writer", "tester", "ethics"]


def test_orchestrator_blends_state_with_prev_state():
    orch = Orchestrator(
        llm=DeterministicProvider(),
        embeddings=HashingEmbeddings(dim=32),
    )
    out1 = orch.run_step({"goal": "first"})
    out2 = orch.run_step({"goal": "second"}, prev_state=out1.state)
    assert len(out2.state) == 32
    # The blended state should not equal the prior state exactly.
    assert out2.state != out1.state


def test_orchestrator_state_dim_matches_embeddings():
    orch = Orchestrator(
        llm=DeterministicProvider(),
        embeddings=HashingEmbeddings(dim=128),
    )
    assert orch.state_dim == 128
    out = orch.run_step({"goal": "x"})
    assert len(out.state) == 128


def test_orchestrator_block_verdict_lowers_e_star():
    block_llm = _ScriptedLLM([
        json.dumps({  # writer
            "assumptions": ["a"], "data": {"sources": ["s"]}, "steps": ["x"],
            "interfaces": ["i"], "acceptance": ["ok"], "risks": ["r"],
        }),
        json.dumps([{"name": "t", "checks": ["c"]}]),  # tester
        json.dumps({"concerns": ["bad"], "mitigations": [], "verdict": "block"}),  # ethics
    ])
    approve_llm = _ScriptedLLM([
        json.dumps({
            "assumptions": ["a"], "data": {"sources": ["s"]}, "steps": ["x"],
            "interfaces": ["i"], "acceptance": ["ok"], "risks": ["r"],
        }),
        json.dumps([{"name": "t", "checks": ["c"]}]),
        json.dumps({"concerns": [], "mitigations": [], "verdict": "approve"}),
    ])
    embed = HashingEmbeddings(dim=32)
    block_step = Orchestrator(llm=block_llm, embeddings=embed).run_step({"goal": "g"})
    approve_step = Orchestrator(llm=approve_llm, embeddings=embed).run_step({"goal": "g"})
    assert block_step.e_star < approve_step.e_star
    assert block_step.ethics["verdict"] == "block"
    assert approve_step.ethics["verdict"] == "approve"
