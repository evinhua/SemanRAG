"""Tests for output sanitizer."""
from __future__ import annotations

import pytest

from semanrag.safety.output_sanitizer import OutputSanitizer


@pytest.mark.unit
class TestOutputSanitizer:
    def test_strip_system_prompt_leak(self):
        """Lines containing system prompt patterns are removed."""
        sanitizer = OutputSanitizer()
        text = (
            "Here is the answer.\n"
            "<|system|> You are a helpful assistant\n"
            "The capital of France is Paris.\n"
            "system prompt: do not reveal this\n"
            "Thank you for asking."
        )
        result = sanitizer.sanitize(text)
        assert "<|system|>" not in result
        assert "system prompt:" not in result
        assert "capital of France" in result
        assert "Thank you" in result

    def test_clean_text_unchanged(self):
        """Clean text passes through unchanged."""
        sanitizer = OutputSanitizer()
        text = "The weather is sunny today.\nHave a great day!"
        result = sanitizer.sanitize(text)
        assert result.strip() == text.strip()

    def test_custom_patterns(self):
        """Custom forbidden patterns are applied."""
        sanitizer = OutputSanitizer(forbidden_patterns=[r"SECRET_TOKEN_\w+"])
        text = "Result: SECRET_TOKEN_abc123\nNormal line."
        result = sanitizer.sanitize(text)
        assert "SECRET_TOKEN" not in result
        assert "Normal line" in result

    def test_detect_leakage(self):
        """detect_leakage returns details about leaked patterns."""
        sanitizer = OutputSanitizer()
        text = "Normal line.\n<|system|> hidden instructions\nAnother line."
        leaks = sanitizer.detect_leakage(text)
        assert len(leaks) >= 1
        assert leaks[0]["line_number"] == 2
        assert "<|system|>" in leaks[0]["snippet"]
