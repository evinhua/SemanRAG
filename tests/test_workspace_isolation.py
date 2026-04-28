"""Tests for workspace isolation."""
from __future__ import annotations

import pytest

from semanrag.base import StorageNameSpace


@pytest.mark.unit
class TestWorkspaceIsolation:
    def test_different_workspaces_isolated(self):
        """Data in ws1 not visible in ws2 — namespaces differ."""
        config = {}
        ns1 = StorageNameSpace(config, namespace="entities", workspace="ws1")
        ns2 = StorageNameSpace(config, namespace="entities", workspace="ws2")
        assert ns1.full_namespace != ns2.full_namespace
        assert ns1.full_namespace == "ws1/entities"
        assert ns2.full_namespace == "ws2/entities"

    def test_same_workspace_shared(self):
        """Data visible within same workspace."""
        config = {}
        ns_a = StorageNameSpace(config, namespace="entities", workspace="shared_ws")
        ns_b = StorageNameSpace(config, namespace="entities", workspace="shared_ws")
        assert ns_a.full_namespace == ns_b.full_namespace
        assert ns_a.full_namespace == "shared_ws/entities"

    def test_no_workspace_uses_namespace_only(self):
        """Without workspace, full_namespace is just the namespace."""
        config = {}
        ns = StorageNameSpace(config, namespace="entities", workspace=None)
        assert ns.full_namespace == "entities"
