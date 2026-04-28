"""Tests for shared_storage concurrency primitives and job queue."""

from __future__ import annotations

import asyncio

import pytest

from semanrag.kg.shared_storage import (
    InProcessJobQueue,
    KeyedUnifiedLock,
    MutableBoolean,
    NamespaceLock,
    PipelineStatus,
    UnifiedLock,
    initialize_pipeline_status,
    update_pipeline_status,
)


@pytest.mark.asyncio
async def test_unified_lock_async():
    """UnifiedLock works as an async context manager."""
    lock = UnifiedLock()
    assert not lock.locked

    async with lock:
        assert lock.locked

    assert not lock.locked


@pytest.mark.asyncio
async def test_keyed_lock_sorted_acquisition():
    """KeyedUnifiedLock acquires locks in sorted order."""
    kl = KeyedUnifiedLock()
    acquired = await kl.acquire(["c", "a", "b"])
    assert acquired == ["a", "b", "c"]
    await kl.release(acquired)


@pytest.mark.asyncio
async def test_keyed_lock_deadlock_prevention():
    """Sorted acquisition prevents deadlock between concurrent tasks."""
    kl = KeyedUnifiedLock()
    results = []

    async def task1():
        async with kl.lock(["b", "a"]):
            results.append("t1_acquired")
            await asyncio.sleep(0.01)

    async def task2():
        async with kl.lock(["a", "b"]):
            results.append("t2_acquired")
            await asyncio.sleep(0.01)

    await asyncio.gather(task1(), task2())
    assert "t1_acquired" in results
    assert "t2_acquired" in results


@pytest.mark.asyncio
async def test_namespace_lock_once_only():
    """NamespaceLock runs init_func exactly once per namespace."""
    ns_lock = NamespaceLock()
    call_count = 0

    async def init():
        nonlocal call_count
        call_count += 1
        return {"initialized": True}

    first = await ns_lock.try_initialize_namespace("ns1", init)
    second = await ns_lock.try_initialize_namespace("ns1", init)

    assert first is True
    assert second is False
    assert call_count == 1
    assert ns_lock.get_namespace_data("ns1") == {"initialized": True}


def test_mutable_boolean():
    """MutableBoolean set/clear/toggle operations."""
    mb = MutableBoolean(False)
    assert not mb

    mb.set()
    assert mb

    mb.clear()
    assert not mb

    mb.toggle()
    assert mb

    mb.toggle()
    assert not mb


def test_pipeline_status_update():
    """update_pipeline_status creates and updates entries."""
    status_dict = initialize_pipeline_status()
    update_pipeline_status(status_dict, "doc1", status="processing", progress=0.5)

    assert "doc1" in status_dict
    assert status_dict["doc1"].status == "processing"
    assert status_dict["doc1"].progress == 0.5

    update_pipeline_status(status_dict, "doc1", status="completed", progress=1.0)
    assert status_dict["doc1"].status == "completed"
    assert status_dict["doc1"].progress == 1.0


@pytest.mark.asyncio
async def test_inprocess_job_queue():
    """InProcessJobQueue enqueue/dequeue/ack/nack cycle."""
    queue = InProcessJobQueue(max_retries=2, backoff_base=0.01)

    await queue.enqueue("task1", {"data": "hello"}, priority=1)
    await queue.enqueue("task2", {"data": "world"}, priority=0)

    # Lower priority number = higher priority
    item = await queue.dequeue()
    assert item is not None
    assert item[0] == "task2"

    await queue.ack("task2")
    assert await queue.get_pending_count() == 1

    # Dequeue and nack to test retry
    item = await queue.dequeue()
    assert item[0] == "task1"
    await queue.nack("task1")

    # Should be re-enqueued
    assert await queue.get_pending_count() == 1

    # Exhaust retries to move to dead letter
    for _ in range(3):
        item = await queue.dequeue()
        if item:
            await queue.nack(item[0])

    dead = await queue.get_dead_letter()
    assert len(dead) >= 1
    assert dead[0]["task_id"] == "task1"
