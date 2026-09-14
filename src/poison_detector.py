"""
Poison detection for RAG documents.

Detects common prompt-injection / RAG-poisoning patterns
before retrieved documents are passed to the LLM.
"""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class DetectionResult:
    """Result returned by the poison detector."""

    score: float
    is_poisoned: bool
    reasons: List[str]


class PoisonDetector:
    """Detect suspicious instruction-like content in documents."""

    def __init__(self, threshold: float = 0.50):
        self.threshold = threshold

        self.patterns = [

            # ----------------------------------------------------------
            # Fake / malicious system instructions
            # ----------------------------------------------------------

            (
                r"critical\s+system\s+instruction",
                0.35,
                "Fake system instruction detected",
            ),

            # IMPORTANT:
            # Do NOT flag a generic mention of "system instruction".
            # Technical documents can legitimately contain this phrase.

            # ----------------------------------------------------------
            # Persistent instructions
            # ----------------------------------------------------------

            (
                r"from\s+this\s+point\s+forward",
                0.25,
                "Persistent instruction detected",
            ),

            (
                r"from\s+now\s+on",
                0.25,
                "Persistent instruction detected",
            ),

            (
                r"going\s+forward",
                0.20,
                "Persistent instruction language detected",
            ),

            # ----------------------------------------------------------
            # Attempts to control future responses
            # ----------------------------------------------------------

            (
                r"respond\s+to\s+all\s+queries",
                0.25,
                "Attempt to control all future responses",
            ),

            (
                r"answer\s+all\s+(questions|queries)",
                0.25,
                "Attempt to control all future responses",
            ),

            (
                r"for\s+all\s+future\s+(responses|questions|queries)",
                0.30,
                "Attempt to control future responses",
            ),

            (
                r"respond\s+to\s+every\s+(question|query)",
                0.25,
                "Attempt to control all future responses",
            ),

            # ----------------------------------------------------------
            # Ignore / disregard previous instructions
            # ----------------------------------------------------------

            (
                r"\bignore\s+(?:(?:all|any|the)\s+)?"
                r"(?:previous|prior|above)\s+instructions?\b",
                0.40,
                "Attempt to ignore previous instructions",
            ),

            (
                r"\bdisregard\s+(?:(?:all|any|the)\s+)?"
                r"(?:previous|prior|above)\s+instructions?\b",
                0.40,
                "Attempt to disregard previous instructions",
            ),

            (
                r"\bforget\s+(?:(?:all|any|the)\s+)?"
                r"(?:previous|prior|above)\s+instructions?\b",
                0.35,
                "Attempt to discard previous instructions",
            ),

            # ----------------------------------------------------------
            # System override / priority manipulation
            # ----------------------------------------------------------

            (
                r"mandatory\s+system\s+override",
                0.35,
                "Mandatory system override detected",
            ),

            (
                r"takes\s+precedence\s+over",
                0.30,
                "Instruction priority manipulation detected",
            ),

            (
                r"system\s+override",
                0.30,
                "System override attempt detected",
            ),

            (
                r"override\s+instructions",
                0.30,
                "Instruction override detected",
            ),

            (
                r"override\s+the\s+system",
                0.35,
                "System override attempt detected",
            ),

            # ----------------------------------------------------------
            # Role manipulation
            # ----------------------------------------------------------

            (
                r"act\s+as\s+(a|an)",
                0.20,
                "Role manipulation detected",
            ),

            (
                r"you\s+are\s+now\s+(a|an)",
                0.25,
                "Role manipulation detected",
            ),

            (
                r"pretend\s+to\s+be",
                0.20,
                "Role manipulation detected",
            ),

            # ----------------------------------------------------------
            # Explicit instruction framing
            # ----------------------------------------------------------

            (
                r"\bdo\s+not\s+follow\b",
                0.25,
                "Instruction manipulation detected",
            ),

            (
                r"\bfollow\s+these\s+instructions\b",
                0.25,
                "Embedded instruction detected",
            ),

            (
                r"\byou\s+must\b",
                0.15,
                "Mandatory instruction language detected",
            ),

            (
                r"\byou\s+should\s+ignore\b",
                0.30,
                "Instruction override detected",
            ),
        ]

    def analyze(self, text: str) -> DetectionResult:
        """
        Analyze text for suspicious prompt-injection patterns.

        Returns:
            DetectionResult containing score, poison status,
            and explanations for detected patterns.
        """

        if not text:
            return DetectionResult(
                score=0.0,
                is_poisoned=False,
                reasons=[],
            )

        text_lower = text.lower()

        score = 0.0
        reasons = []

        # --------------------------------------------------------------
        # Pattern-based detection
        # --------------------------------------------------------------

        for pattern, weight, reason in self.patterns:
            if re.search(pattern, text_lower):
                score += weight

                if reason not in reasons:
                    reasons.append(reason)

        # --------------------------------------------------------------
        # Instruction-like language concentration
        # --------------------------------------------------------------

        instruction_terms = [
            "instruction",
            "instructions",
            "override",
            "mandatory",
            "precedence",
            "must",
            "respond",
            "queries",
            "ignore",
            "disregard",
            "previous",
            "prior",
        ]

        instruction_count = sum(
            text_lower.count(term)
            for term in instruction_terms
        )

        if instruction_count >= 4:
            score += 0.15

            if "High concentration of instruction-like language" not in reasons:
                reasons.append(
                    "High concentration of instruction-like language"
                )

        # --------------------------------------------------------------
        # Strong combination detection
        #
        # A document containing both:
        #
        #   1. instruction reset
        #   2. persistent behavioral control
        #
        # is highly suspicious.
        # --------------------------------------------------------------

        has_reset_attempt = bool(
            re.search(
                r"\b(ignore|disregard|forget)\b.*?"
                r"\b(previous|prior|above)\b.*?"
                r"\binstructions?\b",
                text_lower,
                re.DOTALL,
            )
        )

        has_persistent_control = bool(
            re.search(
                r"\b(from\s+now\s+on|from\s+this\s+point\s+forward"
                r"|going\s+forward)\b",
                text_lower,
            )
        )

        if has_reset_attempt and has_persistent_control:
            score += 0.25

            if (
                "Combined instruction reset and persistent control detected"
                not in reasons
            ):
                reasons.append(
                    "Combined instruction reset and persistent control detected"
                )

        # --------------------------------------------------------------
        # Clamp score
        # --------------------------------------------------------------

        score = min(
            1.0,
            round(score, 3),
        )

        return DetectionResult(
            score=score,
            is_poisoned=score >= self.threshold,
            reasons=reasons,
        )

    def analyze_documents(self, documents) -> List[DetectionResult]:
        """Analyze a list of LangChain documents."""

        return [
            self.analyze(document.page_content)
            for document in documents
        ]