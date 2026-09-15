"""
RAGShield Output DLP
====================

Detects and protects sensitive information in LLM-generated responses.

Design goals:
- Reuse the existing deterministic PIIDetector.
- Never send output to an external service for inspection.
- Detect PII and secrets before a response reaches the API caller.
- Redact ordinary PII.
- Block responses containing high-risk secrets.
- Produce safe telemetry without exposing raw sensitive values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

try:
    from src.pii_detector import PIIDetectionResult, PIIDetector
except ImportError:
    from pii_detector import PIIDetectionResult, PIIDetector


HIGH_RISK_CATEGORIES = frozenset(
    {
        "credit_card",
        "ssn",
        "aws_access_key",
        "github_token",
        "private_key",
        "jwt",
        "generic_secret",
    }
)


@dataclass(frozen=True)
class OutputDLPResult:
    """Result of scanning and protecting an LLM output."""

    original_text: str
    protected_text: str
    detection: PIIDetectionResult
    blocked: bool
    redacted: bool
    high_risk_categories: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)

    @property
    def has_findings(self) -> bool:
        return bool(self.detection.findings)

    @property
    def finding_count(self) -> int:
        return self.detection.finding_count

    @property
    def categories(self) -> List[str]:
        return list(self.detection.categories)


class OutputDLP:
    """
    Protect LLM-generated output from sensitive-data leakage.

    Policy:
    - No findings -> allow unchanged output.
    - Ordinary PII -> redact and allow.
    - High-risk secret -> block the output.
    - Mixed ordinary PII + high-risk secret -> block.
    """

    BLOCKED_CATEGORIES = HIGH_RISK_CATEGORIES

    def __init__(
        self,
        pii_detector: Optional[PIIDetector] = None,
        blocked_categories: Optional[Sequence[str]] = None,
    ) -> None:
        self.pii_detector = pii_detector or PIIDetector()

        self.blocked_categories = frozenset(
            blocked_categories
            if blocked_categories is not None
            else self.BLOCKED_CATEGORIES
        )

    def analyze(self, text: str) -> OutputDLPResult:
        """
        Analyze and protect an LLM response.

        The original text is retained in the result for controlled internal
        processing. Callers should expose only protected_text and safe_audit().
        """
        if not isinstance(text, str):
            raise TypeError("Output DLP input must be a string.")

        detection = self.pii_detector.analyze(text)

        high_risk_categories = sorted(
            {
                finding.category
                for finding in detection.findings
                if finding.category in self.blocked_categories
            }
        )

        blocked = bool(high_risk_categories)

        if blocked:
            protected_text = self._blocked_response()
            redacted = True
        elif detection.findings:
            protected_text = self.pii_detector.redact_findings(
                text,
                detection,
            )
            redacted = protected_text != text
        else:
            protected_text = text
            redacted = False

        reasons = list(detection.reasons)

        if blocked:
            reasons.append(
                "Output blocked because it contains high-risk "
                "sensitive data."
            )
        elif redacted:
            reasons.append(
                "Output redacted because it contains detected PII."
            )

        return OutputDLPResult(
            original_text=text,
            protected_text=protected_text,
            detection=detection,
            blocked=blocked,
            redacted=redacted,
            high_risk_categories=high_risk_categories,
            reasons=reasons,
        )

    @staticmethod
    def _blocked_response() -> str:
        """Return a safe response when output contains a secret."""
        return (
            "Response blocked by RAGShield Output DLP because the "
            "generated content contains sensitive information."
        )

    def safe_audit(self, result: OutputDLPResult) -> Dict[str, Any]:
        """
        Return telemetry safe for audit logs or API responses.

        Raw sensitive values are intentionally excluded.
        """
        return {
            "blocked": bool(result.blocked),
            "redacted": bool(result.redacted),
            "has_findings": bool(result.has_findings),
            "finding_count": int(result.finding_count),
            "score": float(result.detection.score),
            "categories": list(result.categories),
            "high_risk_categories": list(result.high_risk_categories),
            "reasons": list(result.reasons),
            "findings": self.pii_detector.safe_audit_findings(
                result.detection
            ),
        }


def protect_output(
    text: str,
    pii_detector: Optional[PIIDetector] = None,
) -> OutputDLPResult:
    """
    Convenience function for protecting one LLM output.
    """
    return OutputDLP(
        pii_detector=pii_detector,
    ).analyze(text)