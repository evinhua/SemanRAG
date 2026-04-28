from __future__ import annotations

import os
import shutil
import tempfile
from typing import Callable
from unittest.mock import AsyncMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Command-line options
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-integration", action="store_true", default=False, help="Run integration tests")
    parser.addoption("--keep-artifacts", action="store_true", default=False, help="Keep test artifacts after run")
    parser.addoption("--stress-test", action="store_true", default=False, help="Enable stress-test scenarios")
    parser.addoption("--test-workers", type=int, default=1, help="Number of parallel test workers")
    parser.addoption("--run-regression", action="store_true", default=False, help="Run regression tests")


# ---------------------------------------------------------------------------
# Marker-based auto-skip
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_integration = pytest.mark.skip(reason="pass --run-integration to run")
    skip_regression = pytest.mark.skip(reason="pass --run-regression to run")

    for item in items:
        if "integration" in item.keywords and not config.getoption("--run-integration"):
            item.add_marker(skip_integration)
        if "regression" in item.keywords and not config.getoption("--run-regression"):
            item.add_marker(skip_regression)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_working_dir(request: pytest.FixtureRequest, tmp_path: pytest.TempPathFactory):
    """Create and chdir into a temporary working directory; clean up unless --keep-artifacts."""
    original = os.getcwd()
    work = tmp_path / "work"
    work.mkdir()
    os.chdir(work)
    yield work
    os.chdir(original)
    if not request.config.getoption("--keep-artifacts"):
        shutil.rmtree(work, ignore_errors=True)


@pytest.fixture()
def sample_chunks() -> list[dict]:
    """Return a small list of text chunks for unit tests."""
    return [
        {"id": "c1", "content": "Knowledge graphs represent entities and relations.", "tokens": 8},
        {"id": "c2", "content": "Retrieval-augmented generation grounds LLM output in evidence.", "tokens": 9},
        {"id": "c3", "content": "Semantic search uses dense vector embeddings for similarity.", "tokens": 9},
    ]


@pytest.fixture()
def mock_embedding_func() -> Callable:
    """Return an async callable that produces deterministic 128-d embeddings."""
    rng = np.random.default_rng(42)

    async def _embed(texts: list[str]) -> np.ndarray:
        return rng.standard_normal((len(texts), 128)).astype(np.float32)

    return _embed
