"""Phase 3 — in-process pub/sub used to fan step events out to WebSockets.

The MACP bus from Phase 1 is a *distributed* message bus aimed at agent
coordination. The streaming bus here is intentionally simpler and
narrower: it's a per-process async pub/sub that the API uses to push
fresh ECL steps to any WebSocket clients currently watching a given run.

Why not reuse the MACP broker? Because:

* The WebSocket consumer cares about *one* run, not the global stream.
* We want zero external dependencies on the API hot path.
* WebSockets are inherently process-local; a global, in-memory hub is
  the right granularity.

If/when the API is sharded across processes a Redis pubsub adapter can
slot in behind this same interface.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, Set


class StreamHub:
    """A tiny per-run async pub/sub hub.

    Subscribers get their own bounded :class:`asyncio.Queue`; slow
    clients are dropped (events skipped) rather than blocking the
    publisher, so a stalled WebSocket can't backpressure the API.
    """

    def __init__(self, *, max_queue: int = 64) -> None:
        self._subs: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        self._max_queue = max_queue
        self._lock = asyncio.Lock()

    async def publish(self, run_id: str, event: Dict[str, Any]) -> None:
        """Best-effort fan-out of ``event`` to every subscriber of ``run_id``."""
        async with self._lock:
            queues = list(self._subs.get(run_id, ()))
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop the event for this subscriber; they'll get the next one.
                # The full-queue case implies the client isn't reading fast
                # enough, so silently skipping is safer than blocking.
                continue

    @asynccontextmanager
    async def subscribe(self, run_id: str) -> AsyncIterator[asyncio.Queue]:
        """Async context manager that yields a queue of events for ``run_id``."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue)
        async with self._lock:
            self._subs[run_id].add(queue)
        try:
            yield queue
        finally:
            async with self._lock:
                self._subs[run_id].discard(queue)
                if not self._subs[run_id]:
                    self._subs.pop(run_id, None)

    async def subscriber_count(self, run_id: str) -> int:
        async with self._lock:
            return len(self._subs.get(run_id, ()))


# Module-level default hub. Callers can construct their own for tests.
_default_hub: StreamHub | None = None


def get_default_hub() -> StreamHub:
    global _default_hub
    if _default_hub is None:
        _default_hub = StreamHub()
    return _default_hub


def reset_default_hub() -> None:
    """Test helper: drop the module-level hub so the next call creates a fresh one."""
    global _default_hub
    _default_hub = None


__all__ = ["StreamHub", "get_default_hub", "reset_default_hub"]
