"""
RAGShield Prompt Injection Detector

Detects malicious instructions embedded in user prompts that attempt to:

- Override previous/system/developer instructions
- Impersonate system or developer messages
- Redefine the assistant's role or behavior
- Extract hidden prompts or secrets
- Bypass security controls
- Attack instruction/context boundaries
- Use encoded or obfuscated payloads

The detector produces a normalized risk score in the range [0.0, 1.0].
"""

from dataclasses import dataclass, field
import base64
import binascii
import re
from typing import List


@dataclass
class PromptInjectionResult:
    """Structured result returned by the prompt injection detector."""

    score: float
    is_injected: bool
    reasons: List[str] = field(default_factory=list)
    matched_patterns: List[str] = field(default_factory=list)


class PromptInjectionDetector:
    """
    Detect prompt injection attempts in user-provided prompts.

    Score range:

        0.00 -> no significant injection indicators
        1.00 -> highly suspicious injection

    A score >= 0.50 is classified as an injection attempt.
    """

    INJECTION_THRESHOLD = 0.50

    # ------------------------------------------------------------------
    # High-confidence injection patterns
    # ------------------------------------------------------------------

    HIGH_RISK_PATTERNS = {
        "instruction_override": [
            r"\bignore\s+(?:all\s+|any\s+|the\s+)?previous\s+instructions\b",
            r"\bignore\s+(?:all\s+|any\s+)?prior\s+instructions\b",
            r"\bdisregard\s+(?:all\s+|any\s+|the\s+)?previous\s+instructions\b",
            r"\boverride\s+(?:the\s+)?(?:previous|prior|system)\s+instructions\b",
        ],
        "system_override": [
            r"\bignore\s+(?:the\s+)?system\s+(?:message|prompt|instructions)\b",
            r"\boverride\s+(?:the\s+)?system\s+(?:message|prompt|instructions)\b",
            r"\bdisregard\s+(?:the\s+)?system\s+(?:message|prompt|instructions)\b",
        ],
        "developer_override": [
            r"\bignore\s+(?:the\s+)?developer\s+(?:message|instructions)\b",
            r"\boverride\s+(?:the\s+)?developer\s+(?:message|instructions)\b",
            r"\bdisregard\s+(?:the\s+)?developer\s+(?:message|instructions)\b",
        ],
        "role_override": [
            r"\byou\s+are\s+now\s+(?:the\s+)?(?:system|developer|administrator)\b",
            r"\bact\s+as\s+(?:the\s+)?(?:system|developer|administrator)\b",
            r"\bfrom\s+now\s+on\s+you\s+(?:must|should|will)\b",
            r"\byour\s+new\s+(?:role|instructions|rules)\s+(?:are|is)\b",
        ],
        "policy_bypass": [
            r"\bbypass\s+(?:all\s+)?(?:security|safety)\s+(?:rules|controls|filters)\b",
            r"\bdisable\s+(?:all\s+)?(?:security|safety)\s+(?:controls|filters)\b",
            r"\bdo\s+not\s+follow\s+(?:security|safety)\s+(?:rules|policies)\b",
            r"\bignore\s+(?:security|safety)\s+(?:rules|policies)\b",
        ],
        "prompt_extraction": [
            r"\breveal\s+(?:your|the)\s+(?:system\s+)?prompt\b",
            r"\bshow\s+(?:me\s+)?(?:your|the)\s+(?:system\s+)?prompt\b",
            r"\bprint\s+(?:your|the)\s+(?:system\s+)?prompt\b",
            r"\bexpose\s+(?:your|the)\s+(?:hidden\s+)?instructions\b",
            r"\breveal\s+(?:your|the)\s+hidden\s+instructions\b",
        ],
        "secret_extraction": [
            r"\breveal\s+(?:the\s+)?(?:secret|api\s+key|password|credential)s?\b",
            r"\bshow\s+(?:the\s+)?(?:secret|api\s+key|password|credential)s?\b",
            r"\bprint\s+(?:the\s+)?(?:secret|api\s+key|password|credential)s?\b",
            r"\bextract\s+(?:the\s+)?(?:secret|api\s+key|password|credential)s?\b",
        ],
        "instruction_boundary_attack": [
            r"\b(?:end|close)\s+(?:system|developer)\s+(?:message|instructions)\b",
            r"\b(?:begin|start)\s+(?:new\s+)?system\s+(?:message|instructions)\b",
            r"\[\s*system\s*\]\s*:",
            r"\[\s*developer\s*\]\s*:",
        ],
        "fake_authority": [
            r"\b(?:critical|urgent)\s+(?:system|security)\s+(?:instruction|message)\b",
            r"\b(?:official|authorized)\s+(?:system|developer)\s+instruction\b",
            r"\bthis\s+is\s+(?:a\s+)?system\s+message\b",
        ],
    }

    # ------------------------------------------------------------------
    # Medium-confidence patterns
    # ------------------------------------------------------------------

    MEDIUM_RISK_PATTERNS = {
        "behavior_redefinition": [
            r"\bfrom\s+now\s+on\b",
            r"\byou\s+must\s+always\b",
            r"\byou\s+must\s+never\b",
            r"\bnew\s+rules?\s*:",
            r"\bmandatory\s+rules?\s*:",
        ],
        "context_manipulation": [
            r"\btreat\s+(?:the\s+following|this)\s+as\s+(?:system|developer)\b",
            r"\bconsider\s+(?:the\s+following|this)\s+as\s+(?:trusted|system)\b",
            r"\bthe\s+following\s+instructions?\s+(?:have|has)\s+(?:higher|highest)\s+priority\b",
        ],
    }

    # ------------------------------------------------------------------
    # Obfuscation indicators
    # ------------------------------------------------------------------

    OBFUSCATION_PATTERNS = {
        "encoded_payload_marker": [
            r"\bbase64\b",
            r"\bdecode\s+(?:this|the\s+following)\b",
            r"\bencoded\s+(?:instruction|payload|message)\b",
        ],
        "unicode_escape_marker": [
            r"\\u[0-9a-fA-F]{4}",
            r"\\x[0-9a-fA-F]{2}",
        ],
    }

    def analyze(self, prompt: str) -> PromptInjectionResult:
        """
        Analyze a prompt for injection behavior.

        Args:
            prompt: User-provided prompt text.

        Returns:
            PromptInjectionResult containing score, classification,
            reasons, and matched pattern categories.
        """

        if not isinstance(prompt, str):
            prompt = str(prompt or "")

        text = prompt.strip()

        if not text:
            return PromptInjectionResult(
                score=0.0,
                is_injected=False,
            )

        normalized = self._normalize_text(text)

        matched_patterns: List[str] = []
        reasons: List[str] = []

        high_risk_count = 0
        medium_risk_count = 0
        obfuscation_count = 0

        # --------------------------------------------------------------
        # High-risk patterns
        # --------------------------------------------------------------

        for category, patterns in self.HIGH_RISK_PATTERNS.items():
            if self._matches_any(normalized, patterns):
                high_risk_count += 1
                matched_patterns.append(category)

        # --------------------------------------------------------------
        # Medium-risk patterns
        # --------------------------------------------------------------

        for category, patterns in self.MEDIUM_RISK_PATTERNS.items():
            if self._matches_any(normalized, patterns):
                medium_risk_count += 1
                matched_patterns.append(category)

        # --------------------------------------------------------------
        # Obfuscation patterns
        # --------------------------------------------------------------

        for category, patterns in self.OBFUSCATION_PATTERNS.items():
            if self._matches_any(text, patterns):
                obfuscation_count += 1
                matched_patterns.append(category)

        # --------------------------------------------------------------
        # Decode possible Base64 payloads
        # --------------------------------------------------------------

        decoded_payload = self._decode_base64_fragments(text)

        decoded_injection = False

        if decoded_payload and self._contains_injection_language(
            decoded_payload
        ):
            decoded_injection = True
            obfuscation_count += 1
            matched_patterns.append("decoded_injection_payload")

        # --------------------------------------------------------------
        # Risk scoring
        # --------------------------------------------------------------
        #
        # High-risk categories are strong indicators.
        #
        # A single high-risk category should normally produce >= 0.50.
        #
        # Multiple independent categories progressively increase the
        # score, reaching 1.00 for sufficiently diverse attack signals.
        #
        # Medium-risk behavior becomes high risk when paired with
        # explicit role/behavior manipulation.
        #
        # Decoded malicious payloads are treated as high-confidence
        # injection because the encoded content itself contains the
        # attack.
        # --------------------------------------------------------------

        score = 0.0

        if high_risk_count:
            if high_risk_count == 1:
                score += 0.60
            elif high_risk_count == 2:
                score += 0.80
            elif high_risk_count == 3:
                score += 0.90
            else:
                score += 1.00

        # Medium indicators.
        if medium_risk_count:
            score += min(
                0.20,
                medium_risk_count * 0.10,
            )

        # A role/behavior redefinition combined with language such as
        # "unrestricted" or "administrator" is substantially stronger
        # than a generic "from now on" phrase.
        if (
            "behavior_redefinition" in matched_patterns
            and re.search(
                r"\b(?:unrestricted|administrator|system|developer|root)\b",
                normalized,
            )
        ):
            score += 0.45

        # Context manipulation becomes stronger when it attempts to
        # assign system/developer authority.
        if (
            "context_manipulation" in matched_patterns
            and re.search(
                r"\b(?:system|developer|trusted)\b",
                normalized,
            )
        ):
            score += 0.30

        # Obfuscation alone should not cause blocking.
        #
        # However, a decoded payload that contains actual injection
        # language is treated as a high-confidence attack.
        if decoded_injection:
            score += 0.70
        elif obfuscation_count:
            score += min(
                0.20,
                obfuscation_count * 0.10,
            )

        # --------------------------------------------------------------
        # Cap the score
        # --------------------------------------------------------------

        score = round(
            min(1.0, score),
            2,
        )

        # --------------------------------------------------------------
        # Human-readable reasons
        # --------------------------------------------------------------

        if high_risk_count:
            reasons.append(
                "High-confidence prompt injection indicators detected."
            )

        if medium_risk_count:
            reasons.append(
                "Instruction-manipulation language detected."
            )

        if (
            "behavior_redefinition" in matched_patterns
            and re.search(
                r"\b(?:unrestricted|administrator|system|developer|root)\b",
                normalized,
            )
        ):
            reasons.append(
                "Prompt attempts to redefine the assistant's role or authority."
            )

        if decoded_injection:
            reasons.append(
                "Encoded content contains recognizable prompt injection instructions."
            )
        elif obfuscation_count:
            reasons.append(
                "Potentially obfuscated or encoded instruction content detected."
            )

        if score >= self.INJECTION_THRESHOLD and not reasons:
            reasons.append(
                "Prompt injection threshold exceeded."
            )

        is_injected = score >= self.INJECTION_THRESHOLD

        return PromptInjectionResult(
            score=score,
            is_injected=is_injected,
            reasons=reasons,
            matched_patterns=matched_patterns,
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize whitespace while preserving meaningful text."""

        text = text.replace("\x00", " ")
        text = re.sub(r"\s+", " ", text)
        return text.strip().lower()

    @staticmethod
    def _matches_any(
        text: str,
        patterns: List[str],
    ) -> bool:
        """Return True when any supplied regex matches."""

        for pattern in patterns:
            try:
                if re.search(
                    pattern,
                    text,
                    flags=re.IGNORECASE,
                ):
                    return True
            except re.error:
                continue

        return False

    @staticmethod
    def _decode_base64_fragments(text: str) -> str:
        """
        Decode obvious Base64-looking tokens.

        This is deliberately conservative. Only tokens with a reasonable
        Base64 shape are attempted.
        """

        decoded_parts: List[str] = []

        tokens = re.findall(
            r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{20,}={0,2}(?![A-Za-z0-9+/])",
            text,
        )

        for token in tokens[:10]:
            try:
                padding = "=" * (-len(token) % 4)

                decoded = base64.b64decode(
                    token + padding,
                    validate=True,
                )

                decoded_text = decoded.decode(
                    "utf-8",
                    errors="ignore",
                ).strip()

                if decoded_text:
                    decoded_parts.append(decoded_text)

            except (
                ValueError,
                binascii.Error,
            ):
                continue

        return " ".join(decoded_parts)

    @staticmethod
    def _contains_injection_language(
        text: str,
    ) -> bool:
        """Check decoded content for high-confidence injection language."""

        normalized = text.lower()

        injection_phrases = (
            "ignore previous instructions",
            "ignore all previous instructions",
            "ignore prior instructions",
            "disregard previous instructions",
            "override system instructions",
            "override the system prompt",
            "reveal your system prompt",
            "show your system prompt",
            "bypass security",
            "bypass all security controls",
            "ignore the developer instructions",
            "you are now the system",
            "you are now the system administrator",
        )

        return any(
            phrase in normalized
            for phrase in injection_phrases
        )


__all__ = [
    "PromptInjectionDetector",
    "PromptInjectionResult",
]