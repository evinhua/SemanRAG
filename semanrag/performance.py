"""Performance and caching utilities for SemanRAG."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock

import numpy as np


class EmbeddingCache:
    """Cache embeddings with similarity threshold for query reuse. LRU eviction."""

    def __init__(self, max_size: int = 1024):
        self._cache: OrderedDict[str, tuple[str, np.ndarray]] = OrderedDict()
        self._max_size = max_size
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def get(self, text: str, threshold: float = 0.95) -> np.ndarray | None:
        with self._lock:
            # Exact match first
            if text in self._cache:
                self._cache.move_to_end(text)
                self._hits += 1
                return self._cache[text][1]
            # Similarity search over cached texts
            if not self._cache:
                self._misses += 1
                return None
            # Simple character-level similarity heuristic for near-duplicate queries
            for key, (_, embedding) in self._cache.items():
                if self._text_similarity(text, key) >= threshold:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return embedding
            self._misses += 1
            return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        with self._lock:
            if text in self._cache:
                self._cache.move_to_end(text)
                self._cache[text] = (text, embedding)
            else:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
                self._cache[text] = (text, embedding)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        if a == b:
            return 1.0
        la, lb = len(a), len(b)
        if la == 0 or lb == 0:
            return 0.0
        common = sum(1 for ca, cb in zip(a, b) if ca == cb)
        return common / max(la, lb)


class GraphQueryCache:
    """Cache graph query results with dirty-set invalidation."""

    def __init__(self, max_size: int = 512):
        self._cache: OrderedDict[tuple, dict] = OrderedDict()
        self._max_size = max_size
        self._entity_to_keys: dict[str, set[tuple]] = {}
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: tuple) -> dict | None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, key: tuple, result: dict) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._max_size:
                    evicted_key, _ = self._cache.popitem(last=False)
                    self._remove_entity_refs(evicted_key)
                self._cache[key] = result
            # Track entity references for invalidation
            for entity in result.get("entities", []):
                self._entity_to_keys.setdefault(entity, set()).add(key)

    def invalidate(self, entity_names: list[str]) -> None:
        with self._lock:
            keys_to_remove: set[tuple] = set()
            for name in entity_names:
                keys_to_remove.update(self._entity_to_keys.pop(name, set()))
            for k in keys_to_remove:
                self._cache.pop(k, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._entity_to_keys.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
            }

    def _remove_entity_refs(self, key: tuple) -> None:
        for entity_keys in self._entity_to_keys.values():
            entity_keys.discard(key)


class AdaptiveBatcher:
    """Adaptive batch sizing for embeddings based on rate limit feedback."""

    def __init__(self, initial_size: int = 32, min_size: int = 1, max_size: int = 256):
        self._batch_size = initial_size
        self._min_size = min_size
        self._max_size = max_size
        self._lock = Lock()

    def get_batch_size(self) -> int:
        with self._lock:
            return self._batch_size

    def report_success(self, batch_size: int) -> None:
        with self._lock:
            # Slowly increase toward max
            self._batch_size = min(self._max_size, batch_size + max(1, batch_size // 8))

    def report_rate_limit(self, batch_size: int) -> None:
        with self._lock:
            # Halve on rate limit
            self._batch_size = max(self._min_size, batch_size // 2)


@dataclass
class ConnectionPoolConfig:
    """Configuration for database connection pooling."""

    min_size: int = 2
    max_size: int = 10
    max_idle_time: float = 300.0
    statement_timeout: float = 30.0
    read_replica_urls: list[str] = field(default_factory=list)


class PerformanceMetrics:
    """Collect and expose metrics for Prometheus."""

    def __init__(self):
        self._latencies: dict[str, list[float]] = {}
        self._cache_hits: dict[str, int] = {}
        self._cache_misses: dict[str, int] = {}
        self._queue_depths: list[int] = []
        self._lock = Lock()

    def record_latency(self, operation: str, duration: float) -> None:
        with self._lock:
            self._latencies.setdefault(operation, []).append(duration)

    def record_cache_hit(self, cache_type: str) -> None:
        with self._lock:
            self._cache_hits[cache_type] = self._cache_hits.get(cache_type, 0) + 1

    def record_cache_miss(self, cache_type: str) -> None:
        with self._lock:
            self._cache_misses[cache_type] = self._cache_misses.get(cache_type, 0) + 1

    def record_queue_depth(self, depth: int) -> None:
        with self._lock:
            self._queue_depths.append(depth)

    def get_metrics(self) -> dict:
        with self._lock:
            latency_summary = {}
            for op, durations in self._latencies.items():
                latency_summary[op] = {
                    "count": len(durations),
                    "avg": sum(durations) / len(durations) if durations else 0,
                    "max": max(durations) if durations else 0,
                    "p99": sorted(durations)[int(len(durations) * 0.99)] if durations else 0,
                }
            return {
                "latencies": latency_summary,
                "cache_hits": dict(self._cache_hits),
                "cache_misses": dict(self._cache_misses),
                "queue_depth_latest": self._queue_depths[-1] if self._queue_depths else 0,
            }
