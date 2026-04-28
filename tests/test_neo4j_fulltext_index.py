"""Tests for Neo4j fulltext index creation (mock neo4j driver)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

neo4j = pytest.importorskip("neo4j")


@pytest.fixture()
def mock_neo4j_driver():
    driver = AsyncMock()
    session = AsyncMock()
    result = AsyncMock()
    result.__aiter__ = AsyncMock(return_value=iter([]))
    session.run = AsyncMock(return_value=result)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    driver.session = MagicMock(return_value=session)
    return driver, session


@pytest.mark.asyncio
async def test_fulltext_index_creation(mock_neo4j_driver):
    """Verify fulltext index creation Cypher is executed during initialize."""
    driver, session = mock_neo4j_driver

    with patch("semanrag.kg.neo4j_impl.AsyncGraphDatabase") as mock_agd:
        mock_agd.driver = MagicMock(return_value=driver)

        from semanrag.kg.neo4j_impl import Neo4JStorage

        storage = Neo4JStorage(
            global_config={"neo4j": {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "test"}},
            namespace="test",
            workspace="default",
        )
        storage._driver = driver

        await storage.initialize()

        # Check that fulltext index creation was attempted
        all_calls = [str(c) for c in session.run.call_args_list]
        fulltext_calls = [c for c in all_calls if "FULLTEXT INDEX" in c.upper()]
        assert len(fulltext_calls) > 0


@pytest.mark.asyncio
async def test_search_labels_cypher(mock_neo4j_driver):
    """Verify fulltext index uses workspace label in Cypher."""
    driver, session = mock_neo4j_driver

    with patch("semanrag.kg.neo4j_impl.AsyncGraphDatabase") as mock_agd:
        mock_agd.driver = MagicMock(return_value=driver)

        from semanrag.kg.neo4j_impl import Neo4JStorage

        storage = Neo4JStorage(
            global_config={"neo4j": {"uri": "bolt://localhost:7687", "user": "neo4j", "password": "test"}},
            namespace="test",
            workspace="my_workspace",
        )
        storage._driver = driver

        await storage.initialize()

        all_calls = " ".join(str(c) for c in session.run.call_args_list)
        # Workspace label should appear in the fulltext index definition
        assert "my_workspace" in all_calls
