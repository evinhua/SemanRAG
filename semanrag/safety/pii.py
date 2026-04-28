from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class PIICategory(Enum):
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    SSN = "SSN"
    CREDIT_CARD = "CREDIT_CARD"
    IP_ADDRESS = "IP_ADDRESS"
    PERSON = "PERSON"
    LOCATION = "LOCATION"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    MEDICAL = "MEDICAL"
    FINANCIAL = "FINANCIAL"
    CUSTOM = "CUSTOM"


_PRESIDIO_MAP: dict[str, PIICategory] = {
    "EMAIL_ADDRESS": PIICategory.EMAIL,
    "PHONE_NUMBER": PIICategory.PHONE,
    "US_SSN": PIICategory.SSN,
    "CREDIT_CARD": PIICategory.CREDIT_CARD,
    "IP_ADDRESS": PIICategory.IP_ADDRESS,
    "PERSON": PIICategory.PERSON,
    "LOCATION": PIICategory.LOCATION,
    "DATE_TIME": PIICategory.DATE_OF_BIRTH,
    "MEDICAL_LICENSE": PIICategory.MEDICAL,
    "US_BANK_NUMBER": PIICategory.FINANCIAL,
    "IBAN_CODE": PIICategory.FINANCIAL,
    "US_ITIN": PIICategory.FINANCIAL,
}


@dataclass
class PIIPolicy:
    action: str = "flag"  # 'flag' | 'mask' | 'redact' | 'reject'
    categories: list[PIICategory] = field(default_factory=lambda: list(PIICategory))
    custom_patterns: list[dict] = field(default_factory=list)
    mask_char: str = "*"
    mask_length: int = 4


@dataclass
class PIIFinding:
    category: PIICategory
    start: int
    end: int
    score: float
    text: str
    action_taken: str = ""


class PIIRejectionError(Exception):
    pass


def scan_pii(text: str, policy: PIIPolicy) -> list[PIIFinding]:
    findings: list[PIIFinding] = []
    active_cats = set(policy.categories)

    # Try presidio
    try:
        from presidio_analyzer import AnalyzerEngine

        analyzer = AnalyzerEngine()
        results = analyzer.analyze(text=text, language="en")
        for r in results:
            cat = _PRESIDIO_MAP.get(r.entity_type)
            if cat and cat in active_cats:
                findings.append(
                    PIIFinding(
                        category=cat,
                        start=r.start,
                        end=r.end,
                        score=r.score,
                        text=text[r.start : r.end],
                    )
                )
    except ImportError:
        pass

    # Custom patterns
    for cp in policy.custom_patterns:
        for m in re.finditer(cp["pattern"], text):
            findings.append(
                PIIFinding(
                    category=PIICategory.CUSTOM,
                    start=m.start(),
                    end=m.end(),
                    score=1.0,
                    text=m.group(),
                )
            )

    findings.sort(key=lambda f: f.start)
    return findings


def apply_pii_policy(
    text: str, findings: list[PIIFinding], policy: PIIPolicy
) -> tuple[str, list[PIIFinding]]:
    if policy.action == "reject" and findings:
        raise PIIRejectionError(
            f"PII detected: {len(findings)} finding(s). Policy rejects this input."
        )

    if policy.action == "flag":
        for f in findings:
            f.action_taken = "flagged"
        return text, findings

    # Process replacements in reverse order to preserve positions
    result = text
    for f in sorted(findings, key=lambda f: f.start, reverse=True):
        if policy.action == "mask":
            replacement = policy.mask_char * policy.mask_length
            f.action_taken = "masked"
        else:  # redact
            replacement = ""
            f.action_taken = "redacted"
        result = result[: f.start] + replacement + result[f.end :]

    return result, findings
