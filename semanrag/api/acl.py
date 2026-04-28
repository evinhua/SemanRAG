"""SemanRAG access-control list helpers."""

from __future__ import annotations

from semanrag.base import ACLPolicy


def authorize(user_id: str, user_groups: list[str], acl_policy: ACLPolicy) -> bool:
    """Check whether *user_id* (with *user_groups*) may access a resource."""
    return acl_policy.can_access(user_id, user_groups)


def build_acl_filter(user_id: str, user_groups: list[str]) -> dict:
    """Build a storage-layer ACL filter dict for query-time filtering."""
    return {"user_id": user_id, "user_groups": user_groups}


def resolve_groups(user_id: str) -> list[str]:
    """Return the groups that *user_id* belongs to.

    Stub implementation — replace with a real directory/IdP lookup in production.
    """
    return [f"group:{user_id}"]
