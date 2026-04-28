"""Micro-benchmark for reciprocal_rank_fusion with varying result list sizes."""

import time
import uuid

import pytest

from semanrag.utils import reciprocal_rank_fusion

LIST_SIZES = [10, 100, 1000]


def _generate_result_lists(n: int, num_lists: int = 3) -> list[list[dict]]:
    return [
        [{"id": str(uuid.uuid4()), "score": 1.0 / (i + 1)} for i in range(n)]
        for _ in range(num_lists)
    ]


@pytest.mark.benchmark
@pytest.mark.parametrize("size", LIST_SIZES, ids=["10", "100", "1000"])
def test_reciprocal_rank_fusion(size: int):
    result_lists = _generate_result_lists(size)
    start = time.perf_counter()
    result = reciprocal_rank_fusion(result_lists)
    elapsed = time.perf_counter() - start
    print(f"\nreciprocal_rank_fusion({size} items x 3 lists): {elapsed:.6f}s, {len(result)} merged")


if __name__ == "__main__":
    for size in LIST_SIZES:
        result_lists = _generate_result_lists(size)
        start = time.perf_counter()
        result = reciprocal_rank_fusion(result_lists)
        elapsed = time.perf_counter() - start
        print(f"reciprocal_rank_fusion({size} items x 3 lists): {elapsed:.6f}s, {len(result)} merged")
