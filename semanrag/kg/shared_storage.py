"""Concurrent access management for SemanRAG knowledge graph storage."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. UnifiedLock
# ---------------------------------------------------------------------------


class UnifiedLock:
    """Async-compatible lock supporting both threading and asyncio contexts."""

    def __init__(self) -> None:
        self._async_lock = asyncio.Lock()
        self._thread_lock = threading.Lock()

    @property
    def locked(self) -> bool:
        return self._async_lock.locked() or self._thread_lock.locked()

    # Async context manager
    async def __aenter__(self) -> "UnifiedLock":
        await self._async_lock.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._async_lock.release()

    # Sync context manager
    def __enter__(self) -> "UnifiedLock":
        self._thread_lock.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._thread_lock.release()


# ---------------------------------------------------------------------------
# 2. KeyedUnifiedLock
# ---------------------------------------------------------------------------


class _KeyLockEntry:
    __slots__ = ("lock", "last_used", "acquisition_count", "contention_count")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.last_used: float = time.monotonic()
        self.acquisition_count: int = 0
        self.contention_count: int = 0


class KeyedUnifiedLock:
    """Lock manager keyed by entity/relation names with deadlock prevention."""

    def __init__(self, expiry_seconds: float = 300.0) -> None:
        self._locks: dict[str, _KeyLockEntry] = {}
        self._meta_lock = asyncio.Lock()
        self.expiry_seconds = expiry_seconds

    async def _get_or_create(self, key: str) -> _KeyLockEntry:
        async with self._meta_lock:
            if key not in self._locks:
                self._locks[key] = _KeyLockEntry()
            return self._locks[key]

    async def acquire(self, keys: list[str]) -> list[str]:
        """Acquire locks for all keys in sorted order. Rolls back on failure."""
        sorted_keys = sorted(set(keys))
        acquired: list[str] = []
        try:
            for key in sorted_keys:
                entry = await self._get_or_create(key)
                if entry.lock.locked():
                    entry.contention_count += 1
                await entry.lock.acquire()
                entry.last_used = time.monotonic()
                entry.acquisition_count += 1
                acquired.append(key)
        except Exception:
            # Rollback: release all previously acquired locks
            for acq_key in acquired:
                e = self._locks.get(acq_key)
                if e and e.lock.locked():
                    e.lock.release()
            raise
        return acquired

    async def release(self, keys: list[str]) -> None:
        """Release locks for all keys."""
        for key in keys:
            entry = self._locks.get(key)
            if entry and entry.lock.locked():
                entry.lock.release()

    @asynccontextmanager
    async def lock(self, keys: list[str]):
        """Async context manager that acquires and releases keyed locks."""
        acquired = await self.acquire(keys)
        try:
            yield acquired
        finally:
            await self.release(acquired)

    async def cleanup_expired(self) -> int:
        """Remove locks not used for longer than expiry_seconds. Returns count removed."""
        now = time.monotonic()
        removed = 0
        async with self._meta_lock:
            expired = [
                k
                for k, e in self._locks.items()
                if (now - e.last_used) > self.expiry_seconds and not e.lock.locked()
            ]
            for k in expired:
                del self._locks[k]
                removed += 1
        return removed

    def get_debug_counters(self, key: str) -> dict[str, int]:
        entry = self._locks.get(key)
        if entry is None:
            return {"acquisition_count": 0, "contention_count": 0}
        return {
            "acquisition_count": entry.acquisition_count,
            "contention_count": entry.contention_count,
        }


# ---------------------------------------------------------------------------
# 3. NamespaceLock
# ---------------------------------------------------------------------------


class NamespaceLock:
    """Per-namespace data-initialization tracking."""

    def __init__(self) -> None:
        self._initialized: dict[str, bool] = {}
        self._data: dict[str, dict] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._meta_lock = asyncio.Lock()

    async def try_initialize_namespace(
        self, namespace: str, init_func: Callable
    ) -> bool:
        """Run *init_func* exactly once per namespace. Returns True if init ran."""
        async with self._meta_lock:
            if namespace not in self._locks:
                self._locks[namespace] = asyncio.Lock()

        async with self._locks[namespace]:
            if self._initialized.get(namespace, False):
                return False
            result = init_func()
            if asyncio.iscoroutine(result):
                result = await result
            self._data[namespace] = result if isinstance(result, dict) else {}
            self._initialized[namespace] = True
            return True

    def get_namespace_data(self, namespace: str) -> dict | None:
        if not self._initialized.get(namespace, False):
            return None
        return self._data.get(namespace)

    def is_initialized(self, namespace: str) -> bool:
        return self._initialized.get(namespace, False)


# ---------------------------------------------------------------------------
# 4. MutableBoolean
# ---------------------------------------------------------------------------


class MutableBoolean:
    """Thread-safe mutable boolean flag."""

    def __init__(self, value: bool = False) -> None:
        self._value = value
        self._lock = threading.Lock()

    def set(self) -> None:
        with self._lock:
            self._value = True

    def clear(self) -> None:
        with self._lock:
            self._value = False

    def toggle(self) -> None:
        with self._lock:
            self._value = not self._value

    def __bool__(self) -> bool:
        with self._lock:
            return self._value

    def __repr__(self) -> str:
        return f"MutableBoolean({self._value})"


# ---------------------------------------------------------------------------
# 5. Pipeline status management
# ---------------------------------------------------------------------------


@dataclass
class PipelineStatus:
    doc_id: str
    status: str = "pending"  # pending | processing | completed | failed
    progress: float = 0.0
    stage: str = ""
    error: str = ""
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


_pipeline_status_lock = UnifiedLock()


def initialize_pipeline_status() -> dict[str, PipelineStatus]:
    """Create and return a shared pipeline status dict."""
    return {}


def get_pipeline_status_lock() -> UnifiedLock:
    """Return the shared lock for pipeline status updates."""
    return _pipeline_status_lock


def update_pipeline_status(
    status_dict: dict[str, PipelineStatus], doc_id: str, **kwargs: Any
) -> None:
    """Thread-safe update of a pipeline status entry."""
    with _pipeline_status_lock:
        if doc_id not in status_dict:
            status_dict[doc_id] = PipelineStatus(doc_id=doc_id)
        entry = status_dict[doc_id]
        for k, v in kwargs.items():
            if hasattr(entry, k):
                setattr(entry, k, v)
        entry.updated_at = time.time()


# ---------------------------------------------------------------------------
# 6. Workspace management
# ---------------------------------------------------------------------------

_default_workspace: str = "default"


def get_default_workspace() -> str:
    return _default_workspace


def set_default_workspace(workspace: str) -> None:
    global _default_workspace
    _default_workspace = workspace


# ---------------------------------------------------------------------------
# 7. JobQueueAdapter
# ---------------------------------------------------------------------------


@dataclass(order=True)
class _QueueItem:
    priority: int
    task_id: str = field(compare=False)
    payload: dict = field(compare=False, default_factory=dict)
    retries: int = field(compare=False, default=0)


class JobQueueAdapter(ABC):
    """Abstract base for job queue implementations with at-least-once delivery."""

    @abstractmethod
    async def enqueue(
        self, task_id: str, payload: dict, priority: int = 0
    ) -> None: ...

    @abstractmethod
    async def dequeue(self) -> tuple[str, dict] | None: ...

    @abstractmethod
    async def ack(self, task_id: str) -> None: ...

    @abstractmethod
    async def nack(self, task_id: str) -> None: ...

    @abstractmethod
    async def get_pending_count(self) -> int: ...

    @abstractmethod
    async def get_dead_letter(self) -> list[dict]: ...


class InProcessJobQueue(JobQueueAdapter):
    """In-memory asyncio.PriorityQueue-based job queue."""

    def __init__(self, max_retries: int = 3, backoff_base: float = 1.0) -> None:
        self._queue: asyncio.PriorityQueue[_QueueItem] = asyncio.PriorityQueue()
        self._in_flight: dict[str, _QueueItem] = {}
        self._dead_letter: list[dict] = []
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._lock = asyncio.Lock()

    async def enqueue(
        self, task_id: str, payload: dict, priority: int = 0
    ) -> None:
        item = _QueueItem(priority=priority, task_id=task_id, payload=payload)
        await self._queue.put(item)

    async def dequeue(self) -> tuple[str, dict] | None:
        try:
            item = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
        async with self._lock:
            self._in_flight[item.task_id] = item
        return (item.task_id, item.payload)

    async def ack(self, task_id: str) -> None:
        async with self._lock:
            self._in_flight.pop(task_id, None)

    async def nack(self, task_id: str) -> None:
        async with self._lock:
            item = self._in_flight.pop(task_id, None)
        if item is None:
            return
        item.retries += 1
        if item.retries > self._max_retries:
            self._dead_letter.append(
                {
                    "task_id": item.task_id,
                    "payload": item.payload,
                    "retries": item.retries,
                }
            )
            logger.warning("Task %s moved to dead-letter after %d retries", task_id, item.retries)
        else:
            delay = self._backoff_base * (2 ** (item.retries - 1))
            await asyncio.sleep(delay)
            await self._queue.put(item)

    async def get_pending_count(self) -> int:
        return self._queue.qsize()

    async def get_dead_letter(self) -> list[dict]:
        return list(self._dead_letter)


class CeleryJobQueue(JobQueueAdapter):
    """Celery-backed job queue adapter."""

    def __init__(
        self, celery_app: Any = None, max_retries: int = 3, backoff_base: float = 1.0
    ) -> None:
        self._app = celery_app
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._pending: dict[str, dict] = {}
        self._dead_letter: list[dict] = []
        self._retries: dict[str, int] = {}

    async def enqueue(
        self, task_id: str, payload: dict, priority: int = 0
    ) -> None:
        if self._app is None:
            raise RuntimeError("Celery app not configured")
        self._pending[task_id] = payload
        self._app.send_task(
            "semanrag.tasks.process",
            kwargs={"task_id": task_id, "payload": payload},
            priority=priority,
            task_id=task_id,
        )

    async def dequeue(self) -> tuple[str, dict] | None:
        if not self._pending:
            return None
        task_id = next(iter(self._pending))
        return (task_id, self._pending[task_id])

    async def ack(self, task_id: str) -> None:
        self._pending.pop(task_id, None)
        self._retries.pop(task_id, None)

    async def nack(self, task_id: str) -> None:
        retries = self._retries.get(task_id, 0) + 1
        self._retries[task_id] = retries
        payload = self._pending.pop(task_id, {})
        if retries > self._max_retries:
            self._dead_letter.append(
                {"task_id": task_id, "payload": payload, "retries": retries}
            )
        else:
            await self.enqueue(task_id, payload)

    async def get_pending_count(self) -> int:
        return len(self._pending)

    async def get_dead_letter(self) -> list[dict]:
        return list(self._dead_letter)


class ArqJobQueue(JobQueueAdapter):
    """arq-backed job queue adapter."""

    def __init__(
        self, arq_pool: Any = None, max_retries: int = 3, backoff_base: float = 1.0
    ) -> None:
        self._pool = arq_pool
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._pending: dict[str, dict] = {}
        self._dead_letter: list[dict] = []
        self._retries: dict[str, int] = {}

    async def enqueue(
        self, task_id: str, payload: dict, priority: int = 0
    ) -> None:
        if self._pool is None:
            raise RuntimeError("arq pool not configured")
        self._pending[task_id] = payload
        await self._pool.enqueue_job(
            "process_task", task_id=task_id, payload=payload, _job_id=task_id
        )

    async def dequeue(self) -> tuple[str, dict] | None:
        if not self._pending:
            return None
        task_id = next(iter(self._pending))
        return (task_id, self._pending[task_id])

    async def ack(self, task_id: str) -> None:
        self._pending.pop(task_id, None)
        self._retries.pop(task_id, None)

    async def nack(self, task_id: str) -> None:
        retries = self._retries.get(task_id, 0) + 1
        self._retries[task_id] = retries
        payload = self._pending.pop(task_id, {})
        if retries > self._max_retries:
            self._dead_letter.append(
                {"task_id": task_id, "payload": payload, "retries": retries}
            )
        else:
            await self.enqueue(task_id, payload)

    async def get_pending_count(self) -> int:
        return len(self._pending)

    async def get_dead_letter(self) -> list[dict]:
        return list(self._dead_letter)
