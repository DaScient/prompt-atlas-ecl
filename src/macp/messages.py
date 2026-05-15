"""Pydantic message contracts for the MACP bus."""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    """Canonical roles participating in an ECL run.

    The bus is role-agnostic; these names are advisory and used for routing
    (``macp.<role>.<kind>`` subject convention).
    """

    WRITER = "writer"
    TESTER = "tester"
    ETHICS = "ethics"
    LATENT = "latent"          # GPU latent analysis worker
    ORCHESTRATOR = "orchestrator"
    MEMORY = "memory"          # vector-store retriever / writer


class EventKind(str, Enum):
    """Types of events that can flow across the entanglement bus."""

    # Lifecycle
    RUN_STARTED = "run.started"
    RUN_STEP = "run.step"
    RUN_COMPLETED = "run.completed"

    # Agent contributions
    SPEC_PROPOSED = "spec.proposed"
    TESTS_PROPOSED = "tests.proposed"
    ETHICS_REVIEW = "ethics.review"

    # Shared-state updates
    STATE_UPDATE = "state.update"
    E_STAR = "metrics.e_star"

    # Memory ops
    MEMORY_QUERY = "memory.query"
    MEMORY_RESULT = "memory.result"
    MEMORY_WRITE = "memory.write"


class EntanglementEvent(BaseModel):
    """Envelope carried over the MACP bus.

    ``payload`` is intentionally a free-form dict so each event kind can
    carry the data it needs (spec JSON, test list, state vector, etc.).
    The schema is versioned via ``schema_version`` so downstream consumers
    can evolve independently.
    """

    schema_version: int = 1
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    kind: EventKind
    source: AgentRole
    target: Optional[AgentRole] = None
    timestamp: float = Field(default_factory=lambda: time.time())
    payload: Dict[str, Any] = Field(default_factory=dict)
    correlation_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    def subject(self) -> str:
        """Return the canonical NATS subject for this event.

        Format: ``macp.<run_id>.<source>.<kind>``  (kind keeps its dot,
        e.g. ``macp.r-123.writer.spec.proposed``).
        """
        return f"macp.{self.run_id}.{self.source.value}.{self.kind.value}"

    @staticmethod
    def subject_filter(
        run_id: str = "*",
        source: str = "*",
        kind: str = ">",
    ) -> str:
        """Build a NATS subject filter (``*`` = single token, ``>`` = wildcard tail)."""
        return f"macp.{run_id}.{source}.{kind}"
