"""Tests for PII detection and policy enforcement."""
from __future__ import annotations

import pytest

from semanrag.safety.pii import (
    PIICategory,
    PIIFinding,
    PIIPolicy,
    PIIRejectionError,
    apply_pii_policy,
    scan_pii,
)


def _make_findings(text: str) -> list[PIIFinding]:
    """Create synthetic findings for testing without presidio."""
    findings = []
    import re

    email_pat = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
    for m in email_pat.finditer(text):
        findings.append(PIIFinding(
            category=PIICategory.EMAIL, start=m.start(), end=m.end(),
            score=0.99, text=m.group(),
        ))
    phone_pat = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
    for m in phone_pat.finditer(text):
        findings.append(PIIFinding(
            category=PIICategory.PHONE, start=m.start(), end=m.end(),
            score=0.95, text=m.group(),
        ))
    return sorted(findings, key=lambda f: f.start)


@pytest.mark.unit
class TestPIIPolicies:
    def test_flag_policy(self):
        """Findings returned, text unchanged."""
        text = "Contact john@example.com for info."
        findings = _make_findings(text)
        policy = PIIPolicy(action="flag")
        result_text, result_findings = apply_pii_policy(text, findings, policy)
        assert result_text == text  # unchanged
        assert len(result_findings) == 1
        assert result_findings[0].action_taken == "flagged"

    def test_mask_policy(self):
        """PII replaced with asterisks."""
        text = "Email: john@example.com please."
        findings = _make_findings(text)
        policy = PIIPolicy(action="mask", mask_char="*", mask_length=4)
        result_text, result_findings = apply_pii_policy(text, findings, policy)
        assert "john@example.com" not in result_text
        assert "****" in result_text
        assert result_findings[0].action_taken == "masked"

    def test_redact_policy(self):
        """PII removed entirely."""
        text = "Call 555-123-4567 now."
        findings = _make_findings(text)
        policy = PIIPolicy(action="redact")
        result_text, result_findings = apply_pii_policy(text, findings, policy)
        assert "555-123-4567" not in result_text
        assert result_findings[0].action_taken == "redacted"

    def test_reject_policy(self):
        """Raises PIIRejectionError."""
        text = "Email: test@test.com"
        findings = _make_findings(text)
        policy = PIIPolicy(action="reject")
        with pytest.raises(PIIRejectionError, match="PII detected"):
            apply_pii_policy(text, findings, policy)

    def test_custom_regex_pattern(self):
        """Custom pattern detects project-specific PII."""
        text = "Account ID: ACCT-12345-XYZ"
        policy = PIIPolicy(
            action="flag",
            custom_patterns=[{"pattern": r"ACCT-\d{5}-[A-Z]{3}"}],
        )
        findings = scan_pii(text, policy)
        custom = [f for f in findings if f.category == PIICategory.CUSTOM]
        assert len(custom) == 1
        assert custom[0].text == "ACCT-12345-XYZ"
