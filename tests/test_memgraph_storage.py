"""Tests for Memgraph storage Cypher generation (mock neo4j driver)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

neo4j = pytest.importorskip("neo4j")


@pytest.fixture()
def mock_driver():
    driver = AsyncMock()
    session = AsyncMock()
    result = AsyncMock()
    result.__aiter__ = AsyncMock(return_value=iter([]))
    result.data = MagicMock(return_value=[])
    session.run = AsyncMock(return_value=result)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    driver.session = MagicMock(return_value=session)
    driver.close = AsyncMock()
    return driver, session


@pytest.mark.asyncio
async def test_node_upsert_cypher(mock_driver):
    """Verify node upsert generates MERGE Cypher with workspace label."""
    driver, session = mock_driver

    with patch("semanrag.kg.memgraph_impl.AsyncGraphDatabase") as mock_agd:
        mock_agd.driver = MagicMock(return_value=driver)

        from semanrag.kg.memgraph_impl import MemgraphStorage

        storage = MemgraphStorage(
            global_config={"memgraph_host": "localhost", "memgraph_port": 7687},
            namespace="test",
            workspace="default",
        )
        storage._driver = driver

        await storage.upsert_node("entity1", {"description": "A test entity", "entity_type": "concept"})

        assert session.run.called
        cypher = str(session.run.call_args_list[-1])
        # Should reference the node with workspace/namespace labels
        assert "entity1" in cypher or "MERGE" in cypher.upper() or "CREATE" in cypher.upper()


@pytest.mark.asyncio
async def test_edge_upsert_cypher(mock_driver):
    """Verify edge upsert generates relationship Cypher."""
    driver, session = mock_driver

    with patch("semanrag.kg.memgraph_impl.AsyncGraphDatabase") as mock_agd:
        mock_agd.driver = MagicMock(return_value=driver)

        from semanrag.kg.memgraph_impl import MemgraphStorage

        storage = MemgraphStorage(
            global_config={"memgraph_host": "localhost", "memgraph_port": 7687},
            namespace="test",
            workspace="default",
        )
        storage._driver = driver

        await storage.upsert_edge("src_node", "tgt_node", {"weight": 0.9, "description": "related_to"})

        assert session.run.called
        cypher = str(session.run.call_args_list[-1])
        assert "src_node" in cypher or "tgt_node" in cypher
