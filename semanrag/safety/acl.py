from __future__ import annotations

from semanrag.base import ACLPolicy


def authorize_access(
    user_id: str, user_groups: list[str], acl_policy: ACLPolicy | None
) -> bool:
    if acl_policy is None:
        return True
    return acl_policy.can_access(user_id, user_groups)


def build_storage_filter(user_id: str, user_groups: list[str]) -> dict:
    return {
        "should": [
            {"term": {"acl.public": True}},
            {"term": {"acl.owner": user_id}},
            {"terms": {"acl.visible_to_users": [user_id]}},
            {"terms": {"acl.visible_to_groups": user_groups}},
        ]
    }


def resolve_user_groups(
    user_id: str, group_store: dict | None = None
) -> list[str]:
    if group_store is None:
        return []
    return group_store.get(user_id, [])


def validate_acl_policy(policy: ACLPolicy) -> list[str]:
    errors: list[str] = []
    if not policy.public and not policy.owner:
        errors.append("Non-public policy must have an owner")
    if not policy.public and not (
        policy.visible_to_groups or policy.visible_to_users or policy.owner
    ):
        errors.append(
            "Non-public policy must specify at least one of: owner, visible_to_users, visible_to_groups"
        )
    return errors
