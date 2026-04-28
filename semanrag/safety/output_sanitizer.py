from __future__ import annotations

import re

_DEFAULT_FORBIDDEN: list[str] = [
    r"(?i)<\|?system\|?>",
    r"(?i)<<SYS>>",
    r"(?i)\[INST\]",
    r"(?i)system\s*prompt\s*:",
    r"(?i)you\s+are\s+a\s+(helpful\s+)?assistant",
    r"(?i)^#{0,3}\s*system\s+(instructions|prompt|message)",
    r"(?i)<system_prompt>",
    r"(?i)</system_prompt>",
    r"(?i)<instructions>",
    r"(?i)</instructions>",
    r"(?i)role\s*:\s*(system|assistant|user)",
]


class OutputSanitizer:
    def __init__(self, forbidden_patterns: list[str] | None = None):
        self.forbidden_patterns = forbidden_patterns or _DEFAULT_FORBIDDEN
        self._compiled = [re.compile(p) for p in self.forbidden_patterns]

    def sanitize(self, text: str) -> str:
        lines = text.splitlines(keepends=True)
        return "".join(
            line
            for line in lines
            if not any(p.search(line) for p in self._compiled)
        )

    def detect_leakage(self, text: str) -> list[dict]:
        leaks: list[dict] = []
        for i, line in enumerate(text.splitlines(), start=1):
            for pat, compiled in zip(self.forbidden_patterns, self._compiled):
                if compiled.search(line):
                    leaks.append(
                        {"pattern": pat, "line_number": i, "snippet": line.strip()}
                    )
        return leaks
