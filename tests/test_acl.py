"""Tests for ACL (access control list) logic."""
from __future__ import annotations

import pytest

from semanrag.base import ACLPolicy
from semanrag.safety.acl import authorize_access, build_storage_filter


@pytest.mark.unit
class TestACL:
    def test_public_access(self):
        """Anyone can access public resources."""
        policy = ACLPolicy(public=True)
        assert authorize_access("random_user", [], policy)
        assert authorize_access("", [], policy)

    def test_owner_access(self):
        """Owner can access their own resource."""
        policy = ACLPolicy(public=False, owner="user1")
        assert authorize_access("user1", [], policy)
        assert not authorize_access("user2", [], policy)

    def test_group_access(self):
        """Group member can access."""
        policy = ACLPolicy(public=False, owner="admin", visible_to_groups=["engineering"])
        assert authorize_access("user1", ["engineering"], policy)
        assert authorize_access("user2", ["engineering", "sales"], policy)
        assert not authorize_access("user3", ["marketing"], policy)

    def test_denied_access(self):
        """Non-member denied."""
        policy = ACLPolicy(
            public=False, owner="admin",
            visible_to_users=["user1"],
            visible_to_groups=["team_a"],
        )
        assert not authorize_access("user2", ["team_b"], policy)
        assert authorize_access("user1", [], policy)
        assert authorize_access("user3", ["team_a"], policy)

    def test_none_policy_allows_all(self):
        """None ACL policy means public access."""
        assert authorize_access("anyone", [], None)

    def test_storage_filter_injection(self):
        """Verify storage filter structure for OpenSearch."""
        filt = build_storage_filter("user1", ["group_a", "group_b"])
        assert "should" in filt
        clauses = filt["should"]
        assert len(clauses) == 4
        # Check user_id appears in filter
        user_terms = [c for c in clauses if "term" in c and "acl.owner" in c.get("term", {})]
        assert len(user_terms) == 1
        assert user_terms[0]["term"]["acl.owner"] == "user1"
        # Check groups
        group_terms = [c for c in clauses if "terms" in c and "acl.visible_to_groups" in c.get("terms", {})]
        assert group_terms[0]["terms"]["acl.visible_to_groups"] == ["group_a", "group_b"]
