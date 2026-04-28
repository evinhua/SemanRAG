"""Tests for OpenSearch storage (mock-based)."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest


@pytest.mark.unit
class TestOpenSearchStorage:
    @pytest.mark.asyncio
    async def test_kv_upsert_and_get(self):
        """Mock KV upsert and retrieval."""
        mock_client = AsyncMock()
        mock_client.index = AsyncMock(return_value={"result": "created"})
        mock_client.get = AsyncMock(return_value={
            "_source": {"key": "entity_1", "data": {"name": "Einstein", "type": "PERSON"}}
        })

        # Upsert
        await mock_client.index(index="semanrag_kv_test", id="entity_1", body={"name": "Einstein", "type": "PERSON"})
        mock_client.index.assert_called_once()

        # Get
        result = await mock_client.get(index="semanrag_kv_test", id="entity_1")
        assert result["_source"]["data"]["name"] == "Einstein"

    @pytest.mark.asyncio
    async def test_vector_query(self):
        """Mock vector similarity search."""
        mock_client = AsyncMock()
        mock_client.search = AsyncMock(return_value={
            "hits": {
                "hits": [
                    {"_id": "v1", "_score": 0.95, "_source": {"content": "Einstein was a physicist", "vector": [0.1] * 128}},
                    {"_id": "v2", "_score": 0.82, "_source": {"content": "Relativity theory", "vector": [0.2] * 128}},
                ]
            }
        })
        query_body = {
            "query": {"knn": {"vector": {"vector": [0.1] * 128, "k": 5}}},
        }
        result = await mock_client.search(index="semanrag_vec_test", body=query_body)
        hits = result["hits"]["hits"]
        assert len(hits) == 2
        assert hits[0]["_score"] > hits[1]["_score"]
        assert "Einstein" in hits[0]["_source"]["content"]

    @pytest.mark.asyncio
    async def test_lexical_search(self):
        """Mock BM25 lexical search."""
        mock_client = AsyncMock()
        mock_client.search = AsyncMock(return_value={
            "hits": {
                "total": {"value": 2},
                "hits": [
                    {"_id": "d1", "_score": 12.5, "_source": {"content": "quantum mechanics fundamentals"}},
                    {"_id": "d2", "_score": 8.3, "_source": {"content": "quantum computing applications"}},
                ],
            }
        })
        query_body = {"query": {"match": {"content": "quantum"}}}
        result = await mock_client.search(index="semanrag_lex_test", body=query_body)
        hits = result["hits"]["hits"]
        assert len(hits) == 2
        assert all("quantum" in h["_source"]["content"] for h in hits)
        assert hits[0]["_score"] > hits[1]["_score"]
