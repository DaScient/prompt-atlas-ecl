"""Phase 5 — concrete multi-agent orchestration on top of the MACP bus.

The Phase 1 MACP bus and base ``Agent`` already existed; Phase 5 wires
them up to the new :mod:`src.llm` / :mod:`src.embeddings` modules so
runs can be driven by real LLMs and real embeddings.

The :class:`Orchestrator` is intentionally bus-light — it operates
directly on the agents in-process rather than going through NATS. This
keeps the API hot path synchronous (matching the existing ``Core.step``
contract) and avoids us needing a broker just to run a single step. The
MACP bus is still available for distributed deployments; agents accept
it via dependency injection.
"""

from src.agents.orchestrator import (
    AgentStepResult,
    Orchestrator,
    OrchestratorStep,
)

__all__ = ["AgentStepResult", "Orchestrator", "OrchestratorStep"]
