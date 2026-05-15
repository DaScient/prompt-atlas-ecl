"""Tests for the Phase 3 streaming hub."""
import asyncio

import pytest

from src.streaming import StreamHub, get_default_hub, reset_default_hub


@pytest.mark.asyncio
async def test_subscribe_receives_published_events():
    hub = StreamHub()
    received = []

    async def consumer():
        async with hub.subscribe("r1") as queue:
            for _ in range(2):
                received.append(await queue.get())

    consumer_task = asyncio.create_task(consumer())
    # Yield so subscriber registers before we publish.
    await asyncio.sleep(0)

    await hub.publish("r1", {"t": 1})
    await hub.publish("r1", {"t": 2})

    await asyncio.wait_for(consumer_task, timeout=1.0)
    assert received == [{"t": 1}, {"t": 2}]


@pytest.mark.asyncio
async def test_publish_with_no_subscribers_is_safe():
    hub = StreamHub()
    # Should not raise even with no listeners.
    await hub.publish("nobody", {"t": 99})


@pytest.mark.asyncio
async def test_subscribe_isolates_runs():
    hub = StreamHub()
    got_a, got_b = [], []

    async def consume(run_id, sink):
        async with hub.subscribe(run_id) as queue:
            sink.append(await queue.get())

    ta = asyncio.create_task(consume("a", got_a))
    tb = asyncio.create_task(consume("b", got_b))
    await asyncio.sleep(0)

    await hub.publish("a", {"x": 1})
    await hub.publish("b", {"y": 2})
    await asyncio.wait_for(asyncio.gather(ta, tb), timeout=1.0)

    assert got_a == [{"x": 1}]
    assert got_b == [{"y": 2}]


@pytest.mark.asyncio
async def test_full_queue_is_dropped_not_blocking():
    hub = StreamHub(max_queue=2)

    async with hub.subscribe("slow") as queue:
        # Publisher fires faster than the consumer drains; extras are dropped.
        for i in range(10):
            await hub.publish("slow", {"i": i})
        # Queue holds at most max_queue items; nothing should have raised.
        assert queue.qsize() <= 2


@pytest.mark.asyncio
async def test_subscriber_count_tracks_lifetime():
    hub = StreamHub()
    assert await hub.subscriber_count("r") == 0

    async def hold():
        async with hub.subscribe("r") as _q:
            await asyncio.sleep(0.05)

    t = asyncio.create_task(hold())
    await asyncio.sleep(0.01)
    assert await hub.subscriber_count("r") == 1
    await t
    assert await hub.subscriber_count("r") == 0


def test_default_hub_is_singleton():
    reset_default_hub()
    a = get_default_hub()
    b = get_default_hub()
    assert a is b
    reset_default_hub()
    c = get_default_hub()
    assert c is not a
