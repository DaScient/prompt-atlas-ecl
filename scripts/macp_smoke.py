"""End-to-end smoke test for the Phase 1 MACP + vector-store layer.

Runs entirely in-process using the fallback transports — no NATS or Qdrant
required — so it is safe to execute in CI. When ``NATS_SERVERS`` /
``QDRANT_URL`` env vars are set, the same script exercises the real backends.

    python -m scripts.macp_smoke

The script:
  1. Spins up an :class:`EntanglementBusBroker` (NATS if available, else loopback).
  2. Wires a toy Writer agent that proposes a spec when a run starts.
  3. Wires a toy Tester agent that proposes tests when a spec lands.
  4. Persists the final memory of the run into the vector store.
  5. Recalls a similar memory and prints the round-trip summary.
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import List

from src.macp import (
    Agent,
    AgentRole,
    EntanglementBusBroker,
    EntanglementEvent,
    EventKind,
)
from src.vectorstore import CoLearningMemoryStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("macp.smoke")


class WriterAgent(Agent):
    role = AgentRole.WRITER
    subscribe_kind = EventKind.RUN_STARTED.value

    async def on_event(self, event: EntanglementEvent) -> None:
        spec = {
            "goal": event.payload.get("goal", "explore"),
            "acceptance": ["spec present", "tests present"],
            "risks": ["stub dynamics"],
        }
        await self.emit(EventKind.SPEC_PROPOSED, {"spec": spec})


class TesterAgent(Agent):
    role = AgentRole.TESTER
    subscribe_kind = EventKind.SPEC_PROPOSED.value

    async def on_event(self, event: EntanglementEvent) -> None:
        tests = [{"name": "spec_has_acceptance", "checks": ["acceptance length > 0"]}]
        await self.emit(EventKind.TESTS_PROPOSED, {"tests": tests})


async def main() -> int:
    nats_servers_env = os.getenv("NATS_SERVERS", "").strip()
    nats_servers: List[str] | None = (
        [s.strip() for s in nats_servers_env.split(",") if s.strip()]
        if nats_servers_env
        else None
    )

    broker = EntanglementBusBroker(nats_servers=nats_servers)
    run_id = f"smoke-{uuid.uuid4().hex[:8]}"

    received: list[EntanglementEvent] = []

    async def recorder(event: EntanglementEvent) -> None:
        received.append(event)
        log.info("recv  %-18s  source=%s", event.kind.value, event.source.value)

    async with broker:
        await broker.subscribe(recorder, run_id=run_id)

        writer = WriterAgent(broker, run_id=run_id)
        tester = TesterAgent(broker, run_id=run_id)
        await writer.start()
        await tester.start()

        # Kick off the run.
        start_event = EntanglementEvent(
            run_id=run_id,
            kind=EventKind.RUN_STARTED,
            source=AgentRole.ORCHESTRATOR,
            payload={"goal": "demonstrate MACP roundtrip"},
        )
        await broker.publish(start_event)

        # Give async tasks a moment to fan out.
        await asyncio.sleep(0.25)

    log.info("captured %d events on the bus", len(received))

    # Vector-store roundtrip
    store = CoLearningMemoryStore(
        qdrant_url=os.getenv("QDRANT_URL") or None,
    )
    fake_state = [0.0] * 64
    fake_state[0] = 1.0
    store.remember(
        run_id=run_id,
        vector=fake_state,
        step=1,
        e_star=1.23,
        spec={"goal": "demo"},
        tests=[{"name": "smoke"}],
        tags=["phase1", "smoke"],
    )
    hits = store.recall(fake_state, limit=3)
    log.info("recalled %d memories (top score=%.3f)", len(hits), hits[0][0] if hits else 0.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
