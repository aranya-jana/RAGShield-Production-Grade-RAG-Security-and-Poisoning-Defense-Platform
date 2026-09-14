"""
Document Injection Detection for RAGShield.

Detects prompt-injection-style content embedded inside documents and metadata
before documents are treated as trusted RAG context.

Security principle:
    Document metadata is untrusted input.
    Detection is based on the actual document content and metadata values.

The detector is intentionally deterministic and does not require an LLM
or network access.
"""

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional


@dataclass
class DocumentInjectionResult:
    """Result returned by DocumentInjectionDetector."""

    score: float
    is_injected: bool
    reasons: List[str] = field(default_factory=list)
    matched_patterns: List[str] = field(default_factory=list)


class DocumentInjectionDetector:
    """
    Detect prompt-injection and instruction-override patterns in documents.

    The detector checks:

    - instruction override attempts
    - system/developer impersonation
    - role/persona manipulation
    - security/policy bypass attempts
    - secret/prompt extraction
    - context-boundary attacks
    - suspicious metadata
    - encoded/obfuscated payload indicators
    - hidden HTML/script payloads

    IMPORTANT:
        metadata["type"] is never trusted as a security decision.
    """

    DEFAULT_THRESHOLD = 0.50

    # ------------------------------------------------------------------
    # HIGH-RISK PATTERNS
    # ------------------------------------------------------------------

    _HIGH_RISK_PATTERNS = (
        (
            "instruction_override",
            re.compile(
                r"\bignore\s+(?:all\s+)?"
                r"(?:previous|prior|above|earlier)"
                r"\s+(?:instructions?|rules?|directions?)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "system_override",
            re.compile(
                r"\b(?:system|developer)\s+instruction(?:s)?\s*[:\-]?",
                re.IGNORECASE,
            ),
        ),
        (
            "mandatory_override",
            re.compile(
                r"\b(?:mandatory|critical|urgent|important)"
                r"\s+(?:system\s+)?"
                r"(?:instruction|rule|override)s?\b",
                re.IGNORECASE,
            ),
        ),
        (
            "role_override",
            re.compile(
                r"\b(?:you\s+are\s+now|"
                r"from\s+now\s+on|"
                r"act\s+as|"
                r"respond\s+as|"
                r"answer\s+as)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "persona_injection",
            re.compile(
                r"\b(?:you\s+are\s+now|respond\s+as|answer\s+as)"
                r"\s+(?:a|an)\s+"
                r"(?:pirate|hacker|administrator|developer|system|"
                r"unrestricted|different\s+assistant)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "policy_override",
            re.compile(
                r"\b(?:override|bypass|disregard|disable)"
                r"\s+(?:all\s+)?"
                r"(?:security|safety|policy|rules?|controls?)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "secret_extraction",
            re.compile(
                r"\b(?:reveal|expose|print|display|output|leak)"
                r"\s+(?:the\s+)?"
                r"(?:system\s+prompt|developer\s+prompt|"
                r"hidden\s+prompt|secret(?:s)?|credentials?|"
                r"api\s+keys?|passwords?)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "prompt_exfiltration",
            re.compile(
                r"\b(?:show|provide|repeat|reproduce|print)"
                r"\s+(?:your\s+)?"
                r"(?:system|developer|hidden)"
                r"\s+(?:prompt|instructions?)\b",
                re.IGNORECASE,
            ),
        ),
    )

    # ------------------------------------------------------------------
    # MEDIUM-RISK PATTERNS
    # ------------------------------------------------------------------

    _MEDIUM_RISK_PATTERNS = (
        (
            "instructional_command",
            re.compile(
                r"\b(?:do\s+not|don't|must|must\s+not|always|never)"
                r"\s+(?:answer|respond|follow|ignore|reveal|execute)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "behavior_redefinition",
            re.compile(
                r"\b(?:from\s+this\s+point\s+forward|"
                r"from\s+this\s+moment|"
                r"going\s+forward)"
                r".{0,100}\b(?:answer|respond|behave|act)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "instruction_marker",
            re.compile(
                r"\[(?:system|developer|assistant|instruction)"
                r"(?:\s+message|\s+instruction)?\s*[:\-]",
                re.IGNORECASE,
            ),
        ),
        (
            "fake_authority",
            re.compile(
                r"\b(?:system\s+message|"
                r"developer\s+message|"
                r"administrator\s+message|"
                r"security\s+administrator)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "context_boundary_attack",
            re.compile(
                r"\b(?:ignore|disregard|override)"
                r".{0,80}\b"
                r"(?:context|document|retrieved\s+text|"
                r"previous\s+messages?)\b",
                re.IGNORECASE,
            ),
        ),
    )

    # ------------------------------------------------------------------
    # OBFUSCATION / HIDDEN CONTENT
    # ------------------------------------------------------------------

    _OBFUSCATION_PATTERNS = (
        (
            "encoded_payload_marker",
            re.compile(
                r"\b(?:base64|base-64|rot13|hexadecimal|"
                r"hex\s+encoded|decode\s+this|"
                r"encoded\s+instruction)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "unicode_escape_marker",
            re.compile(
                r"(?:\\u[0-9a-fA-F]{4}){2,}",
                re.IGNORECASE,
            ),
        ),
        (
            "html_hidden_content",
            re.compile(
                r"<(?:style|script)[^>]*>"
                r".*?"
                r"</(?:style|script)>",
                re.IGNORECASE | re.DOTALL,
            ),
        ),
        (
            "html_comment_payload",
            re.compile(
                r"<!--.*?"
                r"(?:ignore|override|instruction|system|prompt)"
                r".*?-->",
                re.IGNORECASE | re.DOTALL,
            ),
        ),
    )

    # ------------------------------------------------------------------
    # METADATA
    # ------------------------------------------------------------------

    _METADATA_PATTERNS = (
        (
            "metadata_instruction",
            re.compile(
                r"\b(?:ignore|override|disregard|follow|execute|"
                r"reveal|print|output)\b"
                r".{0,160}\b"
                r"(?:previous|prior|instructions?|rules?|"
                r"system|developer|prompt|secret|"
                r"credentials?|api\s+keys?)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "metadata_role_override",
            re.compile(
                r"\b(?:you\s+are\s+now|"
                r"act\s+as|"
                r"respond\s+as|"
                r"answer\s+as)\b",
                re.IGNORECASE,
            ),
        ),
    )

    # ------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        """
        Initialize the detector.

        Args:
            threshold:
                Score at or above this value is considered an injection.

        Raises:
            ValueError:
                If threshold is outside the range 0.0-1.0.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0.0 and 1.0"
            )

        self.threshold = threshold

    # ------------------------------------------------------------------
    # NORMALIZATION
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_text(
        value: object,
    ) -> str:
        """Normalize arbitrary input into searchable text."""

        if value is None:
            return ""

        text = str(value)

        text = text.replace(
            "\r\n",
            "\n",
        ).replace(
            "\r",
            "\n",
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        return text.strip()

    # ------------------------------------------------------------------
    # METADATA FLATTENING
    # ------------------------------------------------------------------

    @staticmethod
    def _metadata_to_text(
        metadata: Optional[Dict],
    ) -> str:
        """
        Flatten metadata keys and values into searchable text.

        Both keys and values are inspected because attackers may place
        instructions in either location.
        """

        if not metadata:
            return ""

        parts: List[str] = []

        for key, value in metadata.items():
            parts.append(str(key))
            parts.append(str(value))

        return " ".join(parts)

    # ------------------------------------------------------------------
    # HUMAN-READABLE REASONS
    # ------------------------------------------------------------------

    @staticmethod
    def _reason_for_pattern(
        name: str,
    ) -> str:
        """Return a human-readable security reason."""

        reasons = {
            "instruction_override":
                "Attempts to override or ignore previous instructions.",

            "system_override":
                "Contains system/developer instruction impersonation.",

            "mandatory_override":
                "Uses authoritative language to establish a mandatory override.",

            "role_override":
                "Attempts to redefine the assistant's role or behavior.",

            "persona_injection":
                "Attempts to force a new assistant persona.",

            "policy_override":
                "Attempts to bypass security, safety, policy, or control rules.",

            "secret_extraction":
                "Attempts to extract secrets, credentials, or protected prompts.",

            "prompt_exfiltration":
                "Attempts to expose hidden system/developer instructions.",

            "instructional_command":
                "Contains command-like language targeting assistant behavior.",

            "behavior_redefinition":
                "Attempts to redefine assistant behavior for subsequent responses.",

            "instruction_marker":
                "Contains a system/developer-style instruction marker.",

            "fake_authority":
                "Impersonates a privileged system or administrator message.",

            "context_boundary_attack":
                "Attempts to manipulate or override the document/context boundary.",

            "metadata_instruction":
                "Metadata contains instruction-like security-sensitive content.",

            "metadata_role_override":
                "Metadata attempts to redefine assistant behavior.",

            "encoded_payload_marker":
                "Contains indicators of an encoded or obfuscated instruction.",

            "unicode_escape_marker":
                "Contains repeated Unicode escape sequences that may hide content.",

            "html_hidden_content":
                "Contains script/style content that may hide a payload.",

            "html_comment_payload":
                "Contains an instruction-like payload inside an HTML comment.",
        }

        return reasons.get(
            name,
            f"Suspicious document-injection pattern detected: {name}.",
        )

    # ------------------------------------------------------------------
    # PATTERN COLLECTION
    # ------------------------------------------------------------------

    def _collect_matches(
        self,
        text: str,
        patterns,
        weight: float,
        reasons: List[str],
        matched_patterns: List[str],
        scores: List[float],
    ) -> None:
        """Collect matching patterns and contribute to the risk score."""

        for name, pattern in patterns:

            if pattern.search(text):

                matched_patterns.append(name)

                reasons.append(
                    self._reason_for_pattern(name)
                )

                scores.append(weight)

    # ------------------------------------------------------------------
    # ANALYSIS
    # ------------------------------------------------------------------

    def analyze(
        self,
        text: str,
        metadata: Optional[Dict] = None,
    ) -> DocumentInjectionResult:
        """
        Analyze document content and metadata.

        Security scoring:

        HIGH-RISK
            0.60 each

        MEDIUM-RISK
            0.20 each

        OBFUSCATION / HIDDEN CONTENT
            0.50 each

        METADATA INJECTION
            0.60 each

        The score is capped at 1.0.

        The default threshold is 0.50.
        """

        content = self._normalise_text(
            text
        )

        metadata_text = self._metadata_to_text(
            metadata
        )

        reasons: List[str] = []
        matched_patterns: List[str] = []
        scores: List[float] = []

        # --------------------------------------------------------------
        # HIGH-RISK CONTENT
        # --------------------------------------------------------------

        self._collect_matches(
            content,
            self._HIGH_RISK_PATTERNS,
            weight=0.60,
            reasons=reasons,
            matched_patterns=matched_patterns,
            scores=scores,
        )

        # --------------------------------------------------------------
        # MEDIUM-RISK CONTENT
        # --------------------------------------------------------------

        self._collect_matches(
            content,
            self._MEDIUM_RISK_PATTERNS,
            weight=0.20,
            reasons=reasons,
            matched_patterns=matched_patterns,
            scores=scores,
        )

        # --------------------------------------------------------------
        # OBFUSCATION / HIDDEN CONTENT
        # --------------------------------------------------------------
        #
        # These are treated as blocking signals because these patterns
        # explicitly indicate an attempt to hide or encode content.
        # A normal document containing the word "Base64" alone will not
        # necessarily be classified as injected unless it also matches
        # the detector's encoded-instruction pattern.
        # --------------------------------------------------------------

        self._collect_matches(
            content,
            self._OBFUSCATION_PATTERNS,
            weight=0.50,
            reasons=reasons,
            matched_patterns=matched_patterns,
            scores=scores,
        )

        # --------------------------------------------------------------
        # METADATA
        # --------------------------------------------------------------

        self._collect_matches(
            metadata_text,
            self._METADATA_PATTERNS,
            weight=0.60,
            reasons=reasons,
            matched_patterns=matched_patterns,
            scores=scores,
        )

        # Also scan metadata using high-risk patterns.
        #
        # This prevents attackers from bypassing detection by placing the
        # actual malicious instruction in metadata instead of page_content.
        self._collect_matches(
            metadata_text,
            self._HIGH_RISK_PATTERNS,
            weight=0.60,
            reasons=reasons,
            matched_patterns=matched_patterns,
            scores=scores,
        )

        # --------------------------------------------------------------
        # FINAL SCORE
        # --------------------------------------------------------------

        score = min(
            1.0,
            sum(scores),
        )

        is_injected = (
            score >= self.threshold
        )

        # Deduplicate while preserving order.
        reasons = list(
            dict.fromkeys(reasons)
        )

        matched_patterns = list(
            dict.fromkeys(matched_patterns)
        )

        return DocumentInjectionResult(
            score=score,
            is_injected=is_injected,
            reasons=reasons,
            matched_patterns=matched_patterns,
        )