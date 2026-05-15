"""Tests for the Phase 1 MACP message bus (in-memory transport)."""
from __future__ import annotations

import asyncio

import pytest

from src.macp import (
    Agent,
    AgentRole,
    EntanglementBusBroker,
    EntanglementEvent,
    EventKind,
)
from src.macp.bus import _subject_matches


def test_subject_matcher_single_token_wildcard() -> None:
    assert _subject_matches("macp.*.writer.spec.proposed", "macp.r1.writer.spec.proposed")
    assert not _subject_matches("macp.*.writer.spec.proposed", "macp.r1.tester.spec.proposed")


def test_subject_matcher_tail_wildcard() -> None:
    assert _subject_matches("macp.r1.>", "macp.r1.writer.spec.proposed")
    # `>` requires at least one trailing token.
    assert not _subject_matches("macp.r1.>", "macp.r1")


def test_subject_matcher_exact_length() -> None:
    # Filter without `>` must match token count exactly.
    assert not _subject_matches("macp.r1.writer", "macp.r1.writer.spec")


@pytest.mark.asyncio
async def test_publish_subscribe_roundtrip() -> None:
    broker = EntanglementBusBroker()
    seen: list[EntanglementEvent] = []

    async def handler(event: EntanglementEvent) -> None:
        seen.append(event)

    async with broker:
        await broker.subscribe(handler, run_id="r1")
        await broker.publish(
            EntanglementEvent(
                run_id="r1",
                kind=EventKind.RUN_STARTED,
                source=AgentRole.ORCHESTRATOR,
                payload={"goal": "test"},
            )
        )
        # publish on a different run should NOT be delivered.
        await broker.publish(
            EntanglementEvent(
                run_id="r2",
                kind=EventKind.RUN_STARTED,
                source=AgentRole.ORCHESTRATOR,
            )
        )
        await asyncio.sleep(0.05)

    assert len(seen) == 1
    assert seen[0].run_id == "r1"
    assert seen[0].kind == EventKind.RUN_STARTED


@pytest.mark.asyncio
async def test_agent_ignores_own_emissions() -> None:
    """An Agent must not handle events whose ``source`` matches its own role."""

    calls: list[str] = []

    class Echo(Agent):
        role = AgentRole.WRITER
        subscribe_kind = ">"

        async def on_event(self, event: EntanglementEvent) -> None:
            calls.append(event.kind.value)

    broker = EntanglementBusBroker()
    async with broker:
        agent = Echo(broker, run_id="r1")
        await agent.start()
        # Self-emission: should be ignored.
        await agent.emit(EventKind.SPEC_PROPOSED, {"spec": {}})
        # External emission: should be handled.
        await broker.publish(
            EntanglementEvent(
                run_id="r1",
                kind=EventKind.TESTS_PROPOSED,
                source=AgentRole.TESTER,
            )
        )
        await asyncio.sleep(0.05)

    assert calls == [EventKind.TESTS_PROPOSED.value]
