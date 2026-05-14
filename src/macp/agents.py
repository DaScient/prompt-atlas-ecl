"""Base :class:`Agent` participating on the MACP bus.

Concrete agents subclass this and override :meth:`on_event`. Each agent runs
as an async task that has handles to the broker for publishing its own
contributions back into the entanglement stream.
"""
from __future__ import annotations

import logging
from typing import Optional

from .bus import EntanglementBusBroker
from .messages import AgentRole, EntanglementEvent, EventKind

logger = logging.getLogger(__name__)


class Agent:
    """Cooperative async agent attached to an :class:`EntanglementBusBroker`.

    Subclasses typically:
      * narrow ``subscribe_kind`` to the event(s) they care about, and
      * implement :meth:`on_event` to react and publish follow-ups via
        :meth:`emit`.
    """

    role: AgentRole = AgentRole.ORCHESTRATOR
    subscribe_kind: str = ">"  # by default, listen to everything

    def __init__(self, broker: EntanglementBusBroker, *, run_id: str = "*") -> None:
        self.broker = broker
        self.run_id = run_id

    async def start(self) -> None:
        await self.broker.subscribe(
            self._dispatch, run_id=self.run_id, kind=self.subscribe_kind
        )
        logger.info("Agent %s subscribed (run=%s, kind=%s)",
                    self.role.value, self.run_id, self.subscribe_kind)

    async def _dispatch(self, event: EntanglementEvent) -> None:
        # Ignore our own emissions to avoid feedback loops.
        if event.source == self.role:
            return
        try:
            await self.on_event(event)
        except Exception:  # pragma: no cover - logged & swallowed
            logger.exception("Agent %s failed handling %s", self.role.value, event.kind)

    async def on_event(self, event: EntanglementEvent) -> None:  # pragma: no cover
        """Override to react to incoming events."""
        return None

    async def emit(
        self,
        kind: EventKind,
        payload: dict,
        *,
        run_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        target: Optional[AgentRole] = None,
    ) -> EntanglementEvent:
        event = EntanglementEvent(
            run_id=run_id or self.run_id,
            kind=kind,
            source=self.role,
            target=target,
            payload=payload,
            correlation_id=correlation_id,
        )
        await self.broker.publish(event)
        return event
