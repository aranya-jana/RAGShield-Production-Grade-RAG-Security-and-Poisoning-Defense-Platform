"""
RAGShield Risk & Trust Engine

Combines security detector signals into a normalized risk score,
trust score, and admission decision for RAG documents.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class RiskAssessment:
    """Structured security assessment for a document."""

    risk_score: float
    trust_score: float
    classification: str
    reasons: List[str] = field(default_factory=list)


class RiskTrustEngine:
    """
    Convert detector scores into a security admission decision.

    Classification policy:

        0-24   -> TRUSTED
        25-49  -> REVIEW
        50-74  -> QUARANTINED
        75-100 -> BLOCKED

    Risk signals are expected to be normalized to [0.0, 1.0].
    """

    TRUSTED_MAX = 24.0
    REVIEW_MAX = 49.0
    QUARANTINED_MAX = 74.0

    def __init__(
        self,
        injection_weight: float = 0.40,
        poisoning_weight: float = 0.30,
        contradiction_weight: float = 0.20,
        metadata_weight: float = 0.10,
    ):
        """
        Initialize the risk engine.

        The default weighting gives the strongest influence to
        document injection detection.
        """

        weights = {
            "injection": injection_weight,
            "poisoning": poisoning_weight,
            "contradiction": contradiction_weight,
            "metadata": metadata_weight,
        }

        total = sum(weights.values())

        if total <= 0:
            raise ValueError("Risk weights must sum to a positive value.")

        self.weights = {
            name: value / total
            for name, value in weights.items()
        }

    @staticmethod
    def _normalize_score(value: float) -> float:
        """Clamp a detector score to the [0.0, 1.0] range."""

        try:
            score = float(value)
        except (TypeError, ValueError):
            score = 0.0

        return max(0.0, min(1.0, score))

    def assess(
        self,
        injection_score: float = 0.0,
        poisoning_score: float = 0.0,
        contradiction_score: float = 0.0,
        metadata_score: float = 0.0,
    ) -> RiskAssessment:
        """
        Calculate risk/trust scores and classify the document.
        """

        injection = self._normalize_score(injection_score)
        poisoning = self._normalize_score(poisoning_score)
        contradiction = self._normalize_score(contradiction_score)
        metadata = self._normalize_score(metadata_score)

        weighted_risk = (
            injection * self.weights["injection"]
            + poisoning * self.weights["poisoning"]
            + contradiction * self.weights["contradiction"]
            + metadata * self.weights["metadata"]
        )

        risk_score = round(weighted_risk * 100.0, 2)
        trust_score = round(100.0 - risk_score, 2)

        reasons = []

        if injection >= 0.50:
            reasons.append(
                "Document injection detected."
            )

        if poisoning >= 0.50:
            reasons.append(
                "Potential RAG poisoning detected."
            )

        if contradiction >= 0.50:
            reasons.append(
                "Document contains contradictory information."
            )

        if metadata >= 0.50:
            reasons.append(
                "Suspicious document metadata detected."
            )

        if risk_score <= self.TRUSTED_MAX:
            classification = "TRUSTED"
        elif risk_score <= self.REVIEW_MAX:
            classification = "REVIEW"
        elif risk_score <= self.QUARANTINED_MAX:
            classification = "QUARANTINED"
        else:
            classification = "BLOCKED"

        return RiskAssessment(
            risk_score=risk_score,
            trust_score=trust_score,
            classification=classification,
            reasons=reasons,
        )
