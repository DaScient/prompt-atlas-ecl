"""Multi-agent orchestration: WriterAgent → TesterAgent → EthicsAgent.

Each agent is small and focused: it takes a structured input, asks an
``LLMProvider`` for a completion, and parses the result into a strict
JSON shape (with a deterministic fallback if the LLM returns unparsable
text). The :class:`Orchestrator` chains them and computes a real
embedding-based E-Star proxy from the LLM outputs themselves.

This module is fully optional — it doesn't run unless the new
``PAE_LLM=1`` env is set in ``Core``. The legacy torch stepper remains
the default, so Phase 5 is purely additive.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from src.embeddings.base import EmbeddingProvider
from src.embeddings.hashing import HashingEmbeddings
from src.llm.base import LLMProvider, LLMResponse
from src.llm.deterministic import DeterministicProvider

logger = logging.getLogger(__name__)


# --------------------------------------------------------------- result types


@dataclass
class AgentStepResult:
    """What a single agent produced in one orchestrated step."""

    role: str
    payload: Dict[str, Any]
    llm_provider: str
    confidence: float
    raw_text: str = ""


@dataclass
class OrchestratorStep:
    """Full output of one :meth:`Orchestrator.run_step` invocation."""

    spec: Dict[str, Any]
    tests: List[Dict[str, Any]]
    ethics: Dict[str, Any]
    e_star: float
    state: List[float]
    agents: List[AgentStepResult] = field(default_factory=list)
    divergence: float = 0.0


# ----------------------------------------------------------------- utilities


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def _parse_json(text: str) -> Optional[Any]:
    """Best-effort extraction of a JSON value from an LLM response.

    Handles three formats LLMs commonly produce:

    * pure JSON;
    * JSON inside a fenced ```json …``` block;
    * JSON inside a prose answer (greedy single match).

    Returns ``None`` when nothing parseable is found — callers fall back
    to a deterministic shape so the pipeline never crashes on a bad
    response.
    """
    if not text:
        return None
    s = text.strip()

    # 1) try direct parse
    try:
        return json.loads(s)
    except ValueError:
        pass

    # 2) fenced code block
    m = _FENCED_JSON_RE.search(s)
    if m:
        try:
            return json.loads(m.group(1))
        except ValueError:
            pass

    # 3) bare JSON inside prose
    m = _BARE_JSON_RE.search(s)
    if m:
        try:
            return json.loads(m.group(1))
        except ValueError:
            pass

    return None


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    num = sum(float(a[i]) * float(b[i]) for i in range(n))
    da = math.sqrt(sum(float(x) * float(x) for x in a[:n]))
    db = math.sqrt(sum(float(y) * float(y) for y in b[:n]))
    if da == 0.0 or db == 0.0:
        return 0.0
    return num / (da * db)


# --------------------------------------------------------------------- agents


_WRITER_SYSTEM = (
    "You are the Writer agent in an Entanglement Co-Learning loop. "
    "Read the user's brief and return ONE JSON object with the keys "
    "'assumptions' (list[str]), 'data' (object with 'sources': list[str]), "
    "'steps' (list[str]), 'interfaces' (list[str]), 'acceptance' (list[str]), "
    "and 'risks' (list[str]). Return ONLY the JSON, no prose."
)

_TESTER_SYSTEM = (
    "You are the Tester agent. Given a spec (JSON), return ONE JSON array of "
    "test objects. Each test has 'name' (str) and 'checks' (list[str]). "
    "Cover the acceptance criteria and at least one risk. Return ONLY the JSON."
)

_ETHICS_SYSTEM = (
    "You are the Ethics agent. Given a spec, return ONE JSON object with "
    "'concerns' (list[str]), 'mitigations' (list[str]), and 'verdict' "
    "(one of 'approve', 'revise', 'block'). Be concise. Return ONLY the JSON."
)


def _fallback_spec(brief: Dict[str, Any]) -> Dict[str, Any]:
    goal = brief.get("goal") or brief.get("description") or "unspecified"
    return {
        "assumptions": [f"models co-learn toward: {goal}"],
        "data": {"sources": brief.get("sources", ["synthetic"]) or ["synthetic"]},
        "steps": ["writer: draft spec", "tester: draft tests", "ethics: review", "update: shared state"],
        "interfaces": ["api:/runs/{id}/step"],
        "acceptance": ["spec+tests present", "ethics verdict != block", "E* reported"],
        "risks": ["LLM hallucination", "spec underspecified"],
    }


def _fallback_tests(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"name": "spec_has_acceptance", "checks": ["len(acceptance) > 0"]},
        {"name": "spec_has_risks", "checks": ["len(risks) > 0"]},
    ]


def _fallback_ethics(spec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "concerns": [],
        "mitigations": ["follow the listed risks: " + ", ".join(spec.get("risks", []) or ["n/a"])],
        "verdict": "approve",
    }


class WriterAgent:
    """Drafts a spec from a brief."""

    role = "writer"

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def draft(self, brief: Dict[str, Any]) -> AgentStepResult:
        prompt = "Brief:\n" + json.dumps(brief, indent=2)
        resp = self._llm.complete(prompt, system=_WRITER_SYSTEM, max_tokens=600)
        parsed = _parse_json(resp.text)
        if not isinstance(parsed, dict):
            parsed = _fallback_spec(brief)
        else:
            # Fill any missing required keys with sensible defaults so
            # downstream consumers (Core.step's return shape, dashboard)
            # don't have to defend against partial specs.
            defaults = _fallback_spec(brief)
            for key, default in defaults.items():
                parsed.setdefault(key, default)
        return AgentStepResult(
            role=self.role,
            payload=parsed,
            llm_provider=resp.provider,
            confidence=resp.confidence,
            raw_text=resp.text,
        )


class TesterAgent:
    """Drafts a test list from a spec."""

    role = "tester"
    # Tell pytest this isn't a test class (its "Test" prefix triggers
    # pytest's auto-collection heuristic).
    __test__ = False

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def draft(self, spec: Dict[str, Any]) -> AgentStepResult:
        prompt = "Spec:\n" + json.dumps(spec, indent=2)
        resp = self._llm.complete(prompt, system=_TESTER_SYSTEM, max_tokens=500)
        parsed = _parse_json(resp.text)
        tests: List[Dict[str, Any]]
        if isinstance(parsed, list):
            # Coerce malformed items into the expected shape.
            tests = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                tests.append({
                    "name": str(item.get("name") or f"test_{len(tests)+1}"),
                    "checks": list(item.get("checks") or []),
                })
            if not tests:
                tests = _fallback_tests(spec)
        else:
            tests = _fallback_tests(spec)
        return AgentStepResult(
            role=self.role,
            payload={"tests": tests},
            llm_provider=resp.provider,
            confidence=resp.confidence,
            raw_text=resp.text,
        )


class EthicsAgent:
    """Reviews a spec for ethical / safety concerns."""

    role = "ethics"
    _VALID_VERDICTS = {"approve", "revise", "block"}

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def review(self, spec: Dict[str, Any]) -> AgentStepResult:
        prompt = "Spec:\n" + json.dumps(spec, indent=2)
        resp = self._llm.complete(prompt, system=_ETHICS_SYSTEM, max_tokens=350)
        parsed = _parse_json(resp.text)
        if not isinstance(parsed, dict):
            parsed = _fallback_ethics(spec)
        else:
            parsed.setdefault("concerns", [])
            parsed.setdefault("mitigations", [])
            verdict = str(parsed.get("verdict") or "approve").lower()
            if verdict not in self._VALID_VERDICTS:
                verdict = "approve"
            parsed["verdict"] = verdict
        return AgentStepResult(
            role=self.role,
            payload=parsed,
            llm_provider=resp.provider,
            confidence=resp.confidence,
            raw_text=resp.text,
        )


# ------------------------------------------------------------------- orchestrator


class Orchestrator:
    """Chain Writer → Tester → Ethics and synthesise an E-Star + state.

    The orchestrator is constructed once per process and called
    per-step. It owns LLM + embedding providers but doesn't own any run
    state — that stays in :class:`~src.runstore.base.RunRecord`.
    """

    def __init__(
        self,
        *,
        llm: Optional[LLMProvider] = None,
        embeddings: Optional[EmbeddingProvider] = None,
        writer: Optional[WriterAgent] = None,
        tester: Optional[TesterAgent] = None,
        ethics: Optional[EthicsAgent] = None,
    ) -> None:
        self._llm = llm or DeterministicProvider()
        self._embeddings = embeddings or HashingEmbeddings(dim=256)
        self.writer = writer or WriterAgent(self._llm)
        self.tester = tester or TesterAgent(self._llm)
        self.ethics = ethics or EthicsAgent(self._llm)

    @property
    def state_dim(self) -> int:
        return self._embeddings.dim

    # ------------------------------------------------------------------- core

    def run_step(
        self,
        brief: Dict[str, Any],
        *,
        prev_state: Optional[Sequence[float]] = None,
    ) -> OrchestratorStep:
        """Run one Writer → Tester → Ethics pass and return the step result."""

        # Writer first; everything downstream conditions on its spec.
        w = self.writer.draft(brief)
        spec = w.payload

        # Tester + Ethics could fan out, but they share the same LLM so
        # we run them sequentially to keep behaviour deterministic and
        # rate-limit-friendly.
        t = self.tester.draft(spec)
        e = self.ethics.review(spec)

        # ---- embedding-based latent state ----------------------------
        spec_text = json.dumps(spec, sort_keys=True)
        tests_text = json.dumps(t.payload.get("tests", []), sort_keys=True)
        ethics_text = json.dumps(e.payload, sort_keys=True)

        vecs = self._embeddings.embed([spec_text, tests_text, ethics_text])

        # New state is the centroid of the three agent embeddings; we
        # blend with the previous state to give the system memory across
        # steps (a real co-learning trajectory rather than independent
        # snapshots). 0.5 blend = exponential moving average.
        agent_centroid = _centroid(vecs)
        if prev_state and len(prev_state) == len(agent_centroid):
            new_state = [
                0.5 * float(prev_state[i]) + 0.5 * agent_centroid[i]
                for i in range(len(agent_centroid))
            ]
        else:
            new_state = agent_centroid

        # ---- E-Star proxy --------------------------------------------
        # Higher coherence between writer & tester ⇒ higher E*.
        coherence = _cosine(vecs[0], vecs[1])
        ethics_bonus = 0.0
        verdict = e.payload.get("verdict", "approve")
        if verdict == "approve":
            ethics_bonus = 0.1
        elif verdict == "block":
            ethics_bonus = -0.3
        # Blend the LLM's reported confidences in too.
        llm_conf = (w.confidence + t.confidence + e.confidence) / 3.0
        e_star = max(0.0, min(2.0, 1.0 + coherence + ethics_bonus + 0.4 * (llm_conf - 0.5)))

        # ---- divergence flag (only meaningful for DualLLMProvider) ----
        divergence = 0.0
        for r in (w, t, e):
            # Pull through the dual-LLM divergence if any.
            meta_div = _provider_divergence(r)
            if meta_div is not None:
                divergence = max(divergence, meta_div)

        return OrchestratorStep(
            spec=spec,
            tests=t.payload.get("tests", []),
            ethics=e.payload,
            e_star=float(e_star),
            state=new_state,
            agents=[w, t, e],
            divergence=divergence,
        )


def _centroid(vecs: List[List[float]]) -> List[float]:
    if not vecs:
        return []
    n = len(vecs)
    dim = len(vecs[0])
    return [sum(v[i] for v in vecs) / n for i in range(dim)]


def _provider_divergence(result: AgentStepResult) -> Optional[float]:
    # DualLLMProvider attaches divergence in the response meta; the
    # AgentStepResult only carries the parsed payload + provider name,
    # so this is a best-effort surfacing for the orchestrator step.
    # When real meta forwarding is added (Phase 6 candidate) this becomes
    # exact.
    if result.llm_provider.startswith("dual"):
        # Approximate: if confidence < 0.9 it's likely a contested call.
        return max(0.0, 1.0 - result.confidence)
    return None


__all__ = [
    "AgentStepResult",
    "Orchestrator",
    "OrchestratorStep",
    "WriterAgent",
    "TesterAgent",
    "EthicsAgent",
]
