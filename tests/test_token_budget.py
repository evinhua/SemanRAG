"""Tests for token budget enforcement."""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest


@dataclass
class TokenBudget:
    """Simple token budget tracker for testing."""
    user_limits: dict[str, int] = field(default_factory=dict)
    workspace_limits: dict[str, int] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)

    def set_user_limit(self, user_id: str, limit: int) -> None:
        self.user_limits[user_id] = limit

    def set_workspace_limit(self, workspace: str, limit: int) -> None:
        self.workspace_limits[workspace] = limit

    def record_usage(self, key: str, tokens: int) -> None:
        self.usage[key] = self.usage.get(key, 0) + tokens

    def check_budget(self, key: str, limit_map: dict[str, int]) -> bool:
        """Returns True if within budget."""
        limit = limit_map.get(key)
        if limit is None:
            return True
        return self.usage.get(key, 0) < limit

    def reset(self) -> None:
        self.usage.clear()


@pytest.mark.unit
class TestTokenBudget:
    def test_budget_enforcement_user(self):
        """Exceeds user limit."""
        budget = TokenBudget()
        budget.set_user_limit("user1", 1000)
        budget.record_usage("user1", 1001)
        assert not budget.check_budget("user1", budget.user_limits)

    def test_budget_enforcement_workspace(self):
        """Exceeds workspace limit."""
        budget = TokenBudget()
        budget.set_workspace_limit("ws1", 5000)
        budget.record_usage("ws1", 5001)
        assert not budget.check_budget("ws1", budget.workspace_limits)

    def test_budget_record_and_check(self):
        """Record usage then check."""
        budget = TokenBudget()
        budget.set_user_limit("user1", 1000)
        budget.record_usage("user1", 500)
        assert budget.check_budget("user1", budget.user_limits)
        budget.record_usage("user1", 600)
        assert not budget.check_budget("user1", budget.user_limits)

    def test_budget_reset_daily(self):
        """Reset clears usage."""
        budget = TokenBudget()
        budget.set_user_limit("user1", 1000)
        budget.record_usage("user1", 999)
        assert budget.check_budget("user1", budget.user_limits)
        budget.reset()
        assert budget.usage == {}
        assert budget.check_budget("user1", budget.user_limits)
