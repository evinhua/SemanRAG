"""Tests for description API validation."""
from __future__ import annotations

import pytest


def validate_entity_create(name: str, description: str, entity_type: str = "UNKNOWN") -> list[str]:
    """Validate entity creation request."""
    errors = []
    if not name or not name.strip():
        errors.append("Entity name is required")
    if not description or not description.strip():
        errors.append("Entity description is required")
    if len(name) > 256:
        errors.append("Entity name too long (max 256)")
    return errors


def validate_entity_edit(entity_id: str, description: str | None = None, entity_type: str | None = None) -> list[str]:
    """Validate entity edit request."""
    errors = []
    if not entity_id or not entity_id.strip():
        errors.append("Entity ID is required")
    if description is not None and len(description) > 10000:
        errors.append("Description too long (max 10000)")
    return errors


def validate_relation_create(source: str, target: str, keywords: str = "") -> list[str]:
    """Validate relation creation request."""
    errors = []
    if not source or not source.strip():
        errors.append("Source entity is required")
    if not target or not target.strip():
        errors.append("Target entity is required")
    return errors


@pytest.mark.unit
class TestDescriptionAPIValidation:
    def test_create_entity_empty_description_rejected(self):
        errors = validate_entity_create("Einstein", "")
        assert "Entity description is required" in errors

        errors2 = validate_entity_create("Einstein", "   ")
        assert "Entity description is required" in errors2

    def test_edit_entity_valid(self):
        errors = validate_entity_edit("entity_123", description="Updated description")
        assert errors == []

    def test_create_relation_missing_endpoints(self):
        errors = validate_relation_create("", "target")
        assert "Source entity is required" in errors

        errors2 = validate_relation_create("source", "")
        assert "Target entity is required" in errors2

        errors3 = validate_relation_create("", "")
        assert len(errors3) == 2
