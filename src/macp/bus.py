"""
EntanglementBusBroker — async pub/sub facade over NATS with in-memory fallback.

Design goals
------------
* Zero hard dependency on ``nats-py``: the broker still works in-process so the
  existing API server, unit tests, and the smoke script run without a NATS
  daemon. When ``nats-py`` *is* installed and ``servers`` are reachable we
  transparently upgrade to a real distributed transport.
* JetStream-friendly subject hierarchy (``macp.<run>.<source>.<kind>``) so
  replay/persistence can be enabled by a broker-side stream definition without
  touching application code.
* Strict message contracts via :class:`EntanglementEvent` so consumers can rely
  on schema-version negotiation.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .messages import EntanglementEvent

logger = logging.getLogger(__name__)

# Subscriber callbacks may be sync or async.
Handler = Callable[[EntanglementEvent], Awaitable[None] | None]


# ---------------------------------------------------------------------------
# Transport abstraction
# ---------------------------------------------------------------------------
class _Transport(ABC):
    """Minimal async pub/sub transport interface."""

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def publish(self, subject: str, payload: bytes) -> None: ...

    @abstractmethod
    async def subscribe(
        self, subject_filter: str, cb: Callable[[str, bytes], Awaitable[None]]
    ) -> None: ...


class InMemoryTransport(_Transport):
    """Loopback transport — handy for tests and single-process deployments.

    Matches NATS-style subject wildcards: ``*`` for a single token, ``>`` for
    a multi-token suffix.
    """

    def __init__(self) -> None:
        self._subs: List[tuple[str, Callable[[str, bytes], Awaitable[None]]]] = []
        # Created lazily on first use so the lock binds to the running event
        # loop, not whatever loop happened to be current at construction time.
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def connect(self) -> None:  # no-op
        return None

    async def close(self) -> None:
        async with self._get_lock():
            self._subs.clear()

    async def publish(self, subject: str, payload: bytes) -> None:
        # Snapshot subscribers to avoid mutation during fan-out.
        async with self._get_lock():
            targets = list(self._subs)
        for filt, cb in targets:
            if _subject_matches(filt, subject):
                # Schedule each delivery so a slow handler can't stall others.
                asyncio.create_task(cb(subject, payload))

    async def subscribe(
        self, subject_filter: str, cb: Callable[[str, bytes], Awaitable[None]]
    ) -> None:
        async with self._get_lock():
            self._subs.append((subject_filter, cb))


def _subject_matches(filt: str, subject: str) -> bool:
    """NATS-style subject matcher.

    ``*`` matches exactly one token; ``>`` matches one-or-more remaining tokens
    and must be the last token of the filter.
    """
    f = filt.split(".")
    s = subject.split(".")
    for i, tok in enumerate(f):
        if tok == ">":
            return i < len(s)  # `>` requires at least one more token
        if i >= len(s):
            return False
        if tok == "*":
            continue
        if tok != s[i]:
            return False
    return len(f) == len(s)


class NATSTransport(_Transport):
    """NATS-backed transport. Lazily imports ``nats`` so it stays optional."""

    def __init__(self, servers: List[str], *, name: str = "prompt-atlas-macp") -> None:
        self._servers = servers
        self._name = name
        self._nc: Any = None

    async def connect(self) -> None:
        try:
            import nats  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised only when extra installed
            raise RuntimeError(
                "nats-py is not installed; install with `pip install nats-py` "
                "or use InMemoryTransport"
            ) from exc
        self._nc = await nats.connect(servers=self._servers, name=self._name)
        logger.info("MACP connected to NATS at %s", self._servers)

    async def close(self) -> None:
        if self._nc is not None:
            await self._nc.drain()
            self._nc = None

    async def publish(self, subject: str, payload: bytes) -> None:
        assert self._nc is not None, "transport not connected"
        await self._nc.publish(subject, payload)

    async def subscribe(
        self, subject_filter: str, cb: Callable[[str, bytes], Awaitable[None]]
    ) -> None:
        assert self._nc is not None, "transport not connected"

        async def _wrapper(msg: Any) -> None:  # NATS Msg → our callback
            await cb(msg.subject, msg.data)

        await self._nc.subscribe(subject_filter, cb=_wrapper)


# ---------------------------------------------------------------------------
# Public broker facade
# ---------------------------------------------------------------------------
class EntanglementBusBroker:
    """High-level async pub/sub for :class:`EntanglementEvent` objects.

    Parameters
    ----------
    transport:
        Optional pre-built transport. If omitted, the broker picks NATS when
        ``nats_servers`` is supplied, else falls back to :class:`InMemoryTransport`.
    nats_servers:
        Iterable of NATS URLs (e.g. ``["nats://localhost:4222"]``).
    """

    def __init__(
        self,
        *,
        transport: Optional[_Transport] = None,
        nats_servers: Optional[List[str]] = None,
    ) -> None:
        if transport is not None:
            self._transport: _Transport = transport
        elif nats_servers:
            self._transport = NATSTransport(list(nats_servers))
        else:
            self._transport = InMemoryTransport()
        self._connected = False

    @property
    def transport(self) -> _Transport:
        return self._transport

    async def start(self) -> None:
        if not self._connected:
            await self._transport.connect()
            self._connected = True

    async def stop(self) -> None:
        if self._connected:
            await self._transport.close()
            self._connected = False

    async def publish(self, event: EntanglementEvent) -> None:
        if not self._connected:
            await self.start()
        data = event.model_dump_json().encode("utf-8")
        await self._transport.publish(event.subject(), data)

    async def subscribe(
        self,
        handler: Handler,
        *,
        run_id: str = "*",
        source: str = "*",
        kind: str = ">",
    ) -> None:
        """Register ``handler`` for events matching the filter triple."""
        if not self._connected:
            await self.start()
        subject_filter = EntanglementEvent.subject_filter(run_id, source, kind)

        async def _decode(_subject: str, data: bytes) -> None:
            try:
                event = EntanglementEvent.model_validate_json(data)
            except Exception:  # pragma: no cover - logged & dropped
                logger.exception("MACP failed to decode event on %s", _subject)
                return
            result = handler(event)
            if asyncio.iscoroutine(result):
                await result

        await self._transport.subscribe(subject_filter, _decode)

    # Context-manager sugar so callers can ``async with broker: ...``
    async def __aenter__(self) -> "EntanglementBusBroker":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()
