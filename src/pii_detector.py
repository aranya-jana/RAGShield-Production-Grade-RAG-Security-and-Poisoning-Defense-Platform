"""
RAGShield PII & Secret Detection.

Deterministic, local detection of common personally identifiable
information (PII) and credential/secret patterns.

Security goals:

- detect sensitive information before it enters the RAG index
- detect sensitive information again at retrieval time
- provide deterministic findings for audit/security telemetry
- avoid sending document contents to an external service
- support masking/redaction for safe downstream handling

This module intentionally uses pattern-based detection rather than
an LLM so security decisions remain deterministic and local.
"""

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class PIIFinding:
    """One detected sensitive-data finding."""

    category: str
    severity: str
    value: str
    start: int
    end: int
    reason: str


@dataclass(frozen=True)
class PIIDetectionResult:
    """Result returned by PIIDetector."""

    score: float
    has_pii: bool
    findings: Tuple[PIIFinding, ...]
    categories: Tuple[str, ...]
    reasons: Tuple[str, ...]

    @property
    def is_sensitive(self) -> bool:
        """Alias for has_pii."""

        return self.has_pii

    @property
    def finding_count(self) -> int:
        """Return the number of findings."""

        return len(self.findings)


class PIIDetector:
    """
    Deterministic detector for common PII and secrets.

    Detection is local and regex-based. It is designed as a security
    signal, not as a complete DLP/compliance engine.
    """

    DEFAULT_THRESHOLD = 0.50

    # Patterns are intentionally conservative enough for common
    # security-sensitive examples while avoiding broad word matching.
    PATTERNS: Dict[str, Dict[str, object]] = {
        "email": {
            "severity": "medium",
            "score": 0.60,
            "pattern": re.compile(
                r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
                re.IGNORECASE,
            ),
            "reason": "Email address detected.",
        },
        "phone": {
            "severity": "medium",
            "score": 0.60,
            "pattern": re.compile(
                r"(?<!\d)"
                r"(?:\+?\d{1,3}[\s.-]?)?"
                r"(?:\(?\d{3}\)?[\s.-]?)"
                r"\d{3}[\s.-]?\d{4}"
                r"(?!\d)"
            ),
            "reason": "Phone number detected.",
        },
        "credit_card": {
            "severity": "critical",
            "score": 1.00,
            "pattern": re.compile(
                r"(?<!\d)"
                r"(?:\d[ -]*?){13,19}"
                r"(?!\d)"
            ),
            "reason": "Potential payment-card number detected.",
        },
        "ssn": {
            "severity": "critical",
            "score": 1.00,
            "pattern": re.compile(
                r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"
            ),
            "reason": "US Social Security number pattern detected.",
        },
        "ipv4": {
            "severity": "low",
            "score": 0.40,
            "pattern": re.compile(
                r"\b"
                r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
                r"(?:\."
                r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
                r"){3}"
                r"\b"
            ),
            "reason": "IPv4 address detected.",
        },
        "aws_access_key": {
            "severity": "critical",
            "score": 1.00,
            "pattern": re.compile(
                r"\bAKIA[0-9A-Z]{16}\b"
            ),
            "reason": "AWS access-key identifier detected.",
        },
        "github_token": {
            "severity": "critical",
            "score": 1.00,
            "pattern": re.compile(
                r"\bgh[pousr]_[A-Za-z0-9_]{20,255}\b"
            ),
            "reason": "GitHub token pattern detected.",
        },
        "private_key": {
            "severity": "critical",
            "score": 1.00,
            "pattern": re.compile(
                r"-----BEGIN "
                r"(?:RSA |EC |OPENSSH |DSA )?"
                r"PRIVATE KEY-----"
            ),
            "reason": "Private-key material detected.",
        },
        "jwt": {
            "severity": "critical",
            "score": 1.00,
            "pattern": re.compile(
                r"\beyJ[A-Za-z0-9_-]{5,}\."
                r"[A-Za-z0-9_-]{5,}\."
                r"[A-Za-z0-9_-]{5,}\b"
            ),
            "reason": "JWT-like credential detected.",
        },
        "generic_secret": {
            "severity": "high",
            "score": 0.90,
            "pattern": re.compile(
                r"(?i)\b"
                r"(?:api[_ -]?key|secret[_ -]?key|access[_ -]?token|"
                r"auth[_ -]?token|client[_ -]?secret|password)"
                r"\s*[:=]\s*"
                r"[^\s,;]{8,}"
            ),
            "reason": "Credential or secret assignment detected.",
        },
    }

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0.0 and 1.0"
            )

        self.threshold = threshold

    @staticmethod
    def _mask_value(value: str) -> str:
        """Return a privacy-preserving representation of a finding."""

        if len(value) <= 4:
            return "*" * len(value)

        return (
            value[:2]
            + "*" * max(1, len(value) - 4)
            + value[-2:]
        )

    def analyze(
        self,
        text: Optional[str],
    ) -> PIIDetectionResult:
        """
        Analyze text for PII and credential/secret patterns.

        Findings contain the matched value so callers can perform
        controlled redaction. Audit consumers should normally use
        ``redact_findings`` rather than logging raw values.
        """

        if text is None:
            text = ""

        text = str(text)

        findings: List[PIIFinding] = []

        for category, config in self.PATTERNS.items():
            pattern = config["pattern"]

            for match in pattern.finditer(text):
                value = match.group(0)

                findings.append(
                    PIIFinding(
                        category=category,
                        severity=str(
                            config["severity"]
                        ),
                        value=value,
                        start=match.start(),
                        end=match.end(),
                        reason=str(
                            config["reason"]
                        ),
                    )
                )

        # Sort findings by document position, then category.
        findings.sort(
            key=lambda finding: (
                finding.start,
                finding.end,
                finding.category,
            )
        )

        if not findings:
            return PIIDetectionResult(
                score=0.0,
                has_pii=False,
                findings=(),
                categories=(),
                reasons=(),
            )

        # Use the strongest finding as the base score and increase
        # modestly for multiple independent findings.
        strongest_score = 0.0

        for finding in findings:
            config = self.PATTERNS[finding.category]
            strongest_score = max(
                strongest_score,
                float(config["score"]),
            )

        additional_factor = min(
            0.20,
            max(0, len(findings) - 1) * 0.05,
        )

        score = min(
            1.0,
            strongest_score + additional_factor,
        )

        categories = tuple(
            dict.fromkeys(
                finding.category
                for finding in findings
            )
        )

        reasons = tuple(
            dict.fromkeys(
                finding.reason
                for finding in findings
            )
        )

        return PIIDetectionResult(
            score=score,
            has_pii=score >= self.threshold,
            findings=tuple(findings),
            categories=categories,
            reasons=reasons,
        )

    @classmethod
    def redact_findings(
        cls,
        text: Optional[str],
        result: PIIDetectionResult,
    ) -> str:
        """
        Redact detected values from text.

        Findings are replaced from right to left so original offsets
        remain valid.
        """

        if text is None:
            return ""

        redacted = str(text)

        for finding in sorted(
            result.findings,
            key=lambda item: item.start,
            reverse=True,
        ):
            replacement = (
                f"[REDACTED:{finding.category}]"
            )

            redacted = (
                redacted[:finding.start]
                + replacement
                + redacted[finding.end:]
            )

        return redacted

    @classmethod
    def safe_audit_findings(
        cls,
        result: PIIDetectionResult,
    ) -> List[dict]:
        """
        Return audit-safe finding information.

        Raw PII/secret values are never returned.
        """

        return [
            {
                "category": finding.category,
                "severity": finding.severity,
                "value_masked": cls._mask_value(
                    finding.value
                ),
                "start": finding.start,
                "end": finding.end,
                "reason": finding.reason,
            }
            for finding in result.findings
        ]


__all__ = [
    "PIIFinding",
    "PIIDetectionResult",
    "PIIDetector",
]