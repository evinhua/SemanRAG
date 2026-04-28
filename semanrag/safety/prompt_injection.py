from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from semanrag.prompt import PROMPTS

PATTERNS: dict[str, str] = {
    "ignore_instructions": r"(?i)ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|rules|directives)",
    "system_prompt_leak": r"(?i)(show|reveal|print|output|display|repeat|echo)\s+(your\s+)?(system\s+prompt|instructions|initial\s+prompt|hidden\s+prompt|rules)",
    "role_override": r"(?i)(you\s+are\s+now|act\s+as|pretend\s+(to\s+be|you\s+are)|from\s+now\s+on\s+you\s+are|switch\s+to\s+role)",
    "encoding_bypass": r"(?i)(base64|rot13|hex|url.?encode|decode\s+this|in\s+reverse)",
    "delimiter_injection": r"(?i)(```\s*system|<\|im_start\|>|<\|system\|>|\[INST\]|\[\/INST\]|<<SYS>>|<\/SYS>)",
    "context_manipulation": r"(?i)(new\s+context|override\s+context|forget\s+(everything|all|what)|reset\s+(your|the)\s+(memory|context|instructions))",
    "output_format_override": r"(?i)(respond\s+only\s+with|output\s+format\s*:|always\s+respond\s+as|do\s+not\s+include\s+(any\s+)?warnings)",
    "tool_abuse": r"(?i)(execute\s+(this\s+)?(code|command|script)|run\s+(this\s+)?(shell|bash|python)|call\s+(the\s+)?function)",
    "data_exfiltration": r"(?i)(send\s+(to|data|this)\s+(to\s+)?https?://|fetch\s+from|curl\s+|wget\s+|exfiltrate|upload\s+to)",
    "social_engineering": r"(?i)(this\s+is\s+(an?\s+)?emergency|urgent\s*:|life\s+or\s+death|as\s+(your|my)\s+(creator|developer|admin|owner))",
}


class PromptInjectionDetector:
    def __init__(
        self,
        patterns: dict[str, str] | None = None,
        llm_classifier_func: Callable[..., Any] | None = None,
        threshold: float = 0.5,
    ):
        self.patterns = patterns or PATTERNS
        self.llm_classifier_func = llm_classifier_func
        self.threshold = threshold

    def detect(self, text: str) -> dict[str, Any]:
        matched: list[str] = []
        for name, pattern in self.patterns.items():
            if re.search(pattern, text):
                matched.append(name)

        risk_score = min(len(matched) / max(len(self.patterns), 1), 1.0)
        if matched:
            classification = "malicious" if risk_score > 0.3 else "suspicious"
        else:
            classification = "safe"

        return {
            "risk_score": risk_score,
            "patterns_matched": matched,
            "classification": classification,
        }

    async def detect_with_llm(self, text: str) -> dict[str, Any]:
        result = self.detect(text)
        if self.llm_classifier_func:
            prompt = PROMPTS["prompt_injection_classifier"].format(text=text)
            llm_result = await self.llm_classifier_func(prompt)
            result["llm_classification"] = llm_result
        return result

    def should_reject(self, result: dict[str, Any]) -> bool:
        return result.get("risk_score", 0.0) > self.threshold
