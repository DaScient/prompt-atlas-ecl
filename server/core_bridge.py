# server/core_bridge.py
import logging
import os
from typing import Dict, Any, List, Optional
import torch
from torch import nn

from src.state_bus import EntanglementBus
from src.losses import info_nce
from src.vectorstore import CoLearningMemoryStore

logger = logging.getLogger(__name__)

# Phase 5 imports are lazy/optional — only used when PAE_LLM=1 is set.
try:  # pragma: no cover - exercised by tests via PAE_LLM
    from src.agents import Orchestrator
    from src.llm import get_default_llm
    from src.embeddings import get_default_embeddings
except Exception as _exc:  # noqa: BLE001 - log but never break import
    Orchestrator = None  # type: ignore[assignment]
    get_default_llm = None  # type: ignore[assignment]
    get_default_embeddings = None  # type: ignore[assignment]
    logger.warning("Phase 5 LLM stack unavailable: %s", _exc)


class Core(nn.Module):
    """
    Thin stepper for the API.
    Uses tiny projection heads + GRU bus to advance a shared state and compute an E* proxy.

    When ``PAE_MEMORY=1`` is set, each step is persisted to a
    :class:`CoLearningMemoryStore` (Qdrant if ``QDRANT_URL`` is configured, else
    the in-memory fallback) so future runs can recall past co-learning trajectories.

    When ``PAE_LLM=1`` is set, ``step`` delegates to a Phase 5
    :class:`~src.agents.Orchestrator` that drives Writer/Tester/Ethics
    agents through a real LLM (OpenAI, Anthropic, or the deterministic
    fallback) and computes the latent state from real embeddings. The
    torch path remains the default so legacy callers see unchanged
    behaviour.
    """
    def __init__(
        self,
        device: str = "cpu",
        state_dim: int = 64,
        memory: Optional[CoLearningMemoryStore] = None,
        orchestrator: "Optional[Orchestrator]" = None,
    ):
        super().__init__()
        self.device = device
        self.state_dim = state_dim

        # small projections (stand-ins for pooled LLM hiddens)
        self.projW = nn.Linear(256, 256, bias=False)
        self.projT = nn.Linear(256, 256, bias=False)

        # shared state bus
        self.bus = EntanglementBus(state_dim, in_dim=256 * 2)

        self.to(self.device)

        # Optional long-term memory. Default off to keep behavior unchanged.
        self.memory: Optional[CoLearningMemoryStore] = memory
        if self.memory is None and os.getenv("PAE_MEMORY", "0") == "1":
            self.memory = CoLearningMemoryStore(
                qdrant_url=os.getenv("QDRANT_URL") or None,
                qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,
                vector_size=state_dim,
            )

        # Optional multi-agent orchestrator (Phase 5). Constructed lazily
        # so an import-time failure can never break the torch path.
        self.orchestrator: "Optional[Orchestrator]" = orchestrator
        if (
            self.orchestrator is None
            and os.getenv("PAE_LLM", "0") == "1"
            and Orchestrator is not None
            and get_default_llm is not None
            and get_default_embeddings is not None
        ):
            try:
                self.orchestrator = Orchestrator(
                    llm=get_default_llm(),
                    embeddings=get_default_embeddings(dim=state_dim),
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Orchestrator init failed; using torch path: %s", exc)
                self.orchestrator = None

    @torch.inference_mode()
    def step(
        self,
        S_list: Optional[List[float]] = None,
        *,
        brief: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Phase 5: if an orchestrator is wired up and we have a brief
        # to work from, run the multi-agent pipeline. Otherwise fall back
        # to the legacy torch stepper so existing tests stay green.
        if self.orchestrator is not None and brief is not None:
            try:
                step_out = self.orchestrator.run_step(brief, prev_state=S_list)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Orchestrator.run_step failed; falling back: %s", exc)
            else:
                return {
                    "spec": step_out.spec,
                    "tests": step_out.tests,
                    "e_star": step_out.e_star,
                    "state": step_out.state,
                    "ethics": step_out.ethics,
                    "divergence": step_out.divergence,
                    "agents": [
                        {
                            "role": a.role,
                            "provider": a.llm_provider,
                            "confidence": a.confidence,
                        }
                        for a in step_out.agents
                    ],
                }

        B = 1
        if S_list is None:
            S = torch.zeros(B, self.state_dim, device=self.device)
        else:
            S = torch.tensor(S_list, device=self.device).view(B, -1)

        # pseudo base features (swap with pooled LLM hiddens later)
        baseW = torch.randn(B, 256, device=self.device)
        baseT = torch.randn(B, 256, device=self.device)
        hW = self.projW(baseW)
        hT = self.projT(baseT)

        # advance shared state
        S_next = self.bus(S, hW, hT)

        # E* proxy: invert InfoNCE (lower NCE ⇒ higher coherence)
        L_mi = info_nce(hW, hT, tau=0.1).detach().item()
        e_star = float(max(0.0, 2.0 - L_mi))

        # shaped JSONs
        spec = {
            "assumptions": ["models co-learn via shared state"],
            "data": {"sources": ["synthetic"]},
            "steps": ["writer: draft spec", "tester: draft tests", "update: shared state"],
            "interfaces": ["api:/runs/{id}/step"],
            "acceptance": ["spec+tests present", "E* reported"],
            "risks": ["stub dynamics", "no LLM plugged yet"],
        }
        tests = [
            {"name": "spec_has_acceptance", "checks": ["acceptance length > 0"]},
            {"name": "has_risks", "checks": ["risks length > 0"]},
        ]

        return {
            "spec": spec,
            "tests": tests,
            "e_star": e_star,
            "state": S_next.detach().cpu().view(-1).tolist(),
        }

    def remember_step(
        self,
        *,
        run_id: str,
        step: int,
        state: List[float],
        e_star: float,
        spec: Dict[str, Any],
        tests: List[Dict[str, Any]],
    ) -> None:
        """Persist this step into the long-term memory store, if enabled."""
        if self.memory is None:
            return
        self.memory.remember(
            run_id=run_id,
            step=step,
            vector=list(state),
            e_star=e_star,
            spec=spec,
            tests=tests,
        )
