"""Micro-benchmark for chunking_by_token_size with varying input sizes."""

import time
import pytest

from semanrag.operate import chunking_by_token_size


SIZES = [10_000, 100_000, 1_000_000]


def _generate_text(n: int) -> str:
    base = "The quick brown fox jumps over the lazy dog. "
    repeats = n // len(base) + 1
    return (base * repeats)[:n]


@pytest.mark.benchmark
@pytest.mark.parametrize("size", SIZES, ids=["10k", "100k", "1M"])
def test_chunking_by_token_size(size: int):
    text = _generate_text(size)
    start = time.perf_counter()
    result = chunking_by_token_size(text)
    elapsed = time.perf_counter() - start
    print(f"\nchunking_by_token_size({size} chars): {elapsed:.4f}s, {len(result)} chunks")


if __name__ == "__main__":
    for size in SIZES:
        text = _generate_text(size)
        start = time.perf_counter()
        result = chunking_by_token_size(text)
        elapsed = time.perf_counter() - start
        print(f"chunking_by_token_size({size} chars): {elapsed:.4f}s, {len(result)} chunks")
