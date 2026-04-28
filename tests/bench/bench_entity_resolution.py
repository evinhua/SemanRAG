"""Benchmark entity resolution with mock embeddings at varying scales."""

import time

import numpy as np
import pytest

CANDIDATE_COUNTS = [100, 1000, 10000]


def _entity_resolution_core(label_vecs: list[tuple[str, np.ndarray]], threshold: float = 0.88) -> list[dict]:
    """Core entity resolution logic extracted for benchmarking (avoids async/DB deps)."""
    suggestions = []
    for i in range(len(label_vecs)):
        name_a, vec_a = label_vecs[i]
        norm_a = np.linalg.norm(vec_a)
        if norm_a == 0:
            continue
        for j in range(i + 1, len(label_vecs)):
            name_b, vec_b = label_vecs[j]
            norm_b = np.linalg.norm(vec_b)
            if norm_b == 0:
                continue
            sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
            if sim >= threshold:
                suggestions.append({"entity_a": name_a, "entity_b": name_b, "similarity": sim})
    return suggestions


def _generate_candidates(n: int, dim: int = 384) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(42)
    return [(f"entity_{i}", rng.standard_normal(dim).astype(np.float32)) for i in range(n)]


@pytest.mark.benchmark
@pytest.mark.parametrize("count", CANDIDATE_COUNTS, ids=["100", "1000", "10000"])
def test_entity_resolution(count: int):
    candidates = _generate_candidates(count)
    start = time.perf_counter()
    results = _entity_resolution_core(candidates)
    elapsed = time.perf_counter() - start
    print(f"\nentity_resolution({count} candidates): {elapsed:.4f}s, {len(results)} pairs found")


if __name__ == "__main__":
    for count in CANDIDATE_COUNTS:
        candidates = _generate_candidates(count)
        start = time.perf_counter()
        results = _entity_resolution_core(candidates)
        elapsed = time.perf_counter() - start
        print(f"entity_resolution({count} candidates): {elapsed:.4f}s, {len(results)} pairs found")
