"""
Multi-Agent Communication Protocol (MACP)
=========================================

Asynchronous message bus that lets ECL agents (Writer, Tester, Ethics, etc.)
exchange "entanglement events" across processes / hosts / accelerators.

The default transport is NATS + JetStream when ``nats-py`` is installed and a
broker is reachable; otherwise the broker falls back to an in-process loopback
implementation so unit tests, demos, and the existing API path keep working
without any extra services.

Public surface:
    EntanglementEvent  -- Pydantic message envelope
    AgentRole          -- enum of canonical agent roles
    EntanglementBusBroker
                       -- async publisher/subscriber facade
    Agent              -- minimal base class for role-specific agents
"""

from .messages import AgentRole, EntanglementEvent, EventKind
from .bus import EntanglementBusBroker, InMemoryTransport, NATSTransport
from .agents import Agent

__all__ = [
    "AgentRole",
    "EventKind",
    "EntanglementEvent",
    "EntanglementBusBroker",
    "InMemoryTransport",
    "NATSTransport",
    "Agent",
]
