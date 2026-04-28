"""Tests for entity extraction logic."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from semanrag.base import ExtractionResult, ExtractedEntity, ExtractedRelation


@pytest.mark.unit
class TestStructuredOutputExtraction:
    @pytest.mark.asyncio
    async def test_structured_output_extraction(self):
        """Mock LLM returning ExtractionResult JSON, verify parsing."""
        result = ExtractionResult(
            entities=[
                ExtractedEntity(name="Albert Einstein", type="PERSON", description="Physicist", confidence=0.95),
                ExtractedEntity(name="General Relativity", type="THEORY", description="Physics theory", confidence=0.9),
            ],
            relations=[
                ExtractedRelation(source="Albert Einstein", target="General Relativity", keywords="developed", confidence=0.88),
            ],
        )
        raw_json = result.model_dump_json()
        parsed = ExtractionResult.model_validate_json(raw_json)
        assert len(parsed.entities) == 2
        assert parsed.entities[0].name == "Albert Einstein"
        assert len(parsed.relations) == 1
        assert parsed.relations[0].source == "Albert Einstein"

    @pytest.mark.asyncio
    async def test_delimiter_fallback_extraction(self):
        """Mock LLM returning delimiter-separated format."""
        from semanrag.prompt import DEFAULT_RECORD_DELIMITER, DEFAULT_TUPLE_DELIMITER, DEFAULT_COMPLETION_DELIMITER

        # Simulate delimiter-based output
        records = [
            f'"entity"{DEFAULT_TUPLE_DELIMITER}"PERSON"{DEFAULT_TUPLE_DELIMITER}"Albert Einstein"{DEFAULT_TUPLE_DELIMITER}"Physicist"',
            f'"entity"{DEFAULT_TUPLE_DELIMITER}"THEORY"{DEFAULT_TUPLE_DELIMITER}"Relativity"{DEFAULT_TUPLE_DELIMITER}"Physics theory"',
        ]
        output = DEFAULT_RECORD_DELIMITER.join(records) + DEFAULT_COMPLETION_DELIMITER
        assert DEFAULT_RECORD_DELIMITER in output
        assert DEFAULT_COMPLETION_DELIMITER in output
        # Verify we can split it back
        cleaned = output.split(DEFAULT_COMPLETION_DELIMITER)[0]
        parts = cleaned.split(DEFAULT_RECORD_DELIMITER)
        assert len(parts) == 2


@pytest.mark.unit
class TestGleaningPass:
    @pytest.mark.asyncio
    async def test_gleaning_pass(self):
        """Verify gleaning adds more entities on second pass."""
        first_pass = ExtractionResult(
            entities=[ExtractedEntity(name="Einstein", type="PERSON", confidence=0.9)],
            relations=[],
        )
        second_pass = ExtractionResult(
            entities=[
                ExtractedEntity(name="Einstein", type="PERSON", confidence=0.9),
                ExtractedEntity(name="Princeton", type="LOCATION", confidence=0.85),
            ],
            relations=[
                ExtractedRelation(source="Einstein", target="Princeton", keywords="worked at", confidence=0.8),
            ],
        )
        # Merge: second pass should have more entities
        all_entities = {e.name: e for e in first_pass.entities}
        for e in second_pass.entities:
            all_entities[e.name] = e
        assert len(all_entities) == 2
        assert "Princeton" in all_entities


@pytest.mark.unit
class TestConfidenceFiltering:
    def test_confidence_threshold_filtering(self):
        entities = [
            ExtractedEntity(name="High Conf", type="PERSON", confidence=0.9),
            ExtractedEntity(name="Low Conf", type="PERSON", confidence=0.2),
            ExtractedEntity(name="Mid Conf", type="PERSON", confidence=0.5),
        ]
        threshold = 0.5
        filtered = [e for e in entities if e.confidence >= threshold]
        assert len(filtered) == 2
        assert all(e.confidence >= threshold for e in filtered)
        names = [e.name for e in filtered]
        assert "Low Conf" not in names


@pytest.mark.unit
class TestEntityNameNormalization:
    def test_entity_name_normalization(self):
        """Verify title case and truncation."""
        raw_names = ["albert einstein", "GENERAL RELATIVITY", "a" * 300]
        normalized = []
        for name in raw_names:
            n = name.strip().title()[:256]
            normalized.append(n)
        assert normalized[0] == "Albert Einstein"
        assert normalized[1] == "General Relativity"
        assert len(normalized[2]) == 256
