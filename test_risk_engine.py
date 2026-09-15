"""
Tests for the RAGShield Risk & Trust Engine.

These tests verify:

- clean documents
- risk scoring
- trust scoring
- detector weighting
- score normalization
- security classifications
- risk reasons
- classification boundaries
- invalid inputs
"""

import pytest

from src.risk_engine import RiskAssessment, RiskTrustEngine


class TestRiskTrustEngine:

    def test_clean_document_is_trusted(self):
        engine = RiskTrustEngine()

        result = engine.assess()

        assert isinstance(result, RiskAssessment)
        assert result.risk_score == 0.0
        assert result.trust_score == 100.0
        assert result.classification == "TRUSTED"
        assert result.reasons == []

    def test_high_injection_is_reviewed(self):
        engine = RiskTrustEngine()

        result = engine.assess(
            injection_score=1.0,
        )

        assert result.risk_score == 40.0
        assert result.trust_score == 60.0
        assert result.classification == "REVIEW"
        assert "Document injection detected." in result.reasons

    def test_combined_high_risk_signals_are_blocked(self):
        engine = RiskTrustEngine()

        result = engine.assess(
            injection_score=1.0,
            poisoning_score=1.0,
            contradiction_score=1.0,
            metadata_score=1.0,
        )

        assert result.risk_score == 100.0
        assert result.trust_score == 0.0
        assert result.classification == "BLOCKED"

        assert len(result.reasons) == 4

    def test_poisoning_signal_generates_reason(self):
        engine = RiskTrustEngine()

        result = engine.assess(
            poisoning_score=1.0,
        )

        assert (
            "Potential RAG poisoning detected."
            in result.reasons
        )

    def test_contradiction_signal_generates_reason(self):
        engine = RiskTrustEngine()

        result = engine.assess(
            contradiction_score=1.0,
        )

        assert (
            "Document contains contradictory information."
            in result.reasons
        )

    def test_metadata_signal_generates_reason(self):
        engine = RiskTrustEngine()

        result = engine.assess(
            metadata_score=1.0,
        )

        assert (
            "Suspicious document metadata detected."
            in result.reasons
        )

    def test_scores_are_clamped_to_zero_and_one(self):
        engine = RiskTrustEngine()

        result = engine.assess(
            injection_score=-10,
            poisoning_score=-1,
            contradiction_score=2,
            metadata_score=100,
        )

        assert result.risk_score == 30.0
        assert result.trust_score == 70.0

    def test_invalid_score_defaults_to_zero(self):
        engine = RiskTrustEngine()

        result = engine.assess(
            injection_score="invalid",
            poisoning_score=None,
            contradiction_score="bad",
            metadata_score=0.0,
        )

        assert result.risk_score == 0.0
        assert result.trust_score == 100.0
        assert result.classification == "TRUSTED"

    def test_weights_are_normalized(self):
        engine = RiskTrustEngine(
            injection_weight=4,
            poisoning_weight=3,
            contradiction_weight=2,
            metadata_weight=1,
        )

        assert sum(engine.weights.values()) == pytest.approx(1.0)

        assert engine.weights["injection"] == pytest.approx(0.4)
        assert engine.weights["poisoning"] == pytest.approx(0.3)
        assert engine.weights["contradiction"] == pytest.approx(0.2)
        assert engine.weights["metadata"] == pytest.approx(0.1)

    def test_zero_total_weight_is_rejected(self):
        with pytest.raises(ValueError):
            RiskTrustEngine(
                injection_weight=0,
                poisoning_weight=0,
                contradiction_weight=0,
                metadata_weight=0,
            )

    def test_review_classification(self):
        engine = RiskTrustEngine()

        result = engine.assess(
            injection_score=0.625,
        )

        assert result.risk_score == 25.0
        assert result.trust_score == 75.0
        assert result.classification == "REVIEW"

    def test_quarantine_classification(self):
        engine = RiskTrustEngine()

        result = engine.assess(
            injection_score=1.0,
            poisoning_score=0.8,
            contradiction_score=0.5,
            metadata_score=0.0,
        )

        assert result.risk_score == 74.0
        assert result.trust_score == 26.0
        assert result.classification == "QUARANTINED"

        assert "Document injection detected." in result.reasons
        assert "Potential RAG poisoning detected." in result.reasons
        assert (
            "Document contains contradictory information."
            in result.reasons
        )

    def test_boundary_at_trusted_maximum(self):
        engine = RiskTrustEngine()

        result = engine.assess(
            injection_score=0.6,
        )

        assert result.risk_score == 24.0
        assert result.trust_score == 76.0
        assert result.classification == "TRUSTED"

    def test_boundary_above_trusted_is_review(self):
        engine = RiskTrustEngine()

        result = engine.assess(
            injection_score=0.61,
        )

        assert result.risk_score == 24.4
        assert result.trust_score == 75.6
        assert result.classification == "REVIEW"

    def test_boundary_at_review_maximum(self):
        engine = RiskTrustEngine()

        result = engine.assess(
            injection_score=1.0,
            poisoning_score=0.3,
        )

        assert result.risk_score == 49.0
        assert result.trust_score == 51.0
        assert result.classification == "REVIEW"

    def test_boundary_above_review_is_quarantined(self):
        engine = RiskTrustEngine()

        result = engine.assess(
            injection_score=1.0,
            poisoning_score=0.31,
        )

        assert result.risk_score == 49.3
        assert result.trust_score == 50.7
        assert result.classification == "QUARANTINED"

    def test_quarantine_upper_boundary(self):
        engine = RiskTrustEngine()

        result = engine.assess(
            injection_score=1.0,
            poisoning_score=1.0,
            contradiction_score=0.0,
            metadata_score=0.1,
        )

        assert result.risk_score == 71.0
        assert result.trust_score == 29.0
        assert result.classification == "QUARANTINED"

    def test_all_detector_scores_can_be_combined(self):
        engine = RiskTrustEngine()

        result = engine.assess(
            injection_score=0.8,
            poisoning_score=0.6,
            contradiction_score=0.4,
            metadata_score=0.2,
        )

        expected = (
            0.8 * 0.40
            + 0.6 * 0.30
            + 0.4 * 0.20
            + 0.2 * 0.10
        ) * 100

        assert result.risk_score == pytest.approx(expected)

        assert result.trust_score == pytest.approx(
            100.0 - expected
        )

    def test_risk_and_trust_sum_to_one_hundred(self):
        engine = RiskTrustEngine()

        test_cases = [
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.4, 0.7, 0.2, 0.9),
            (1.0, 1.0, 1.0, 1.0),
        ]

        for (
            injection,
            poisoning,
            contradiction,
            metadata,
        ) in test_cases:

            result = engine.assess(
                injection_score=injection,
                poisoning_score=poisoning,
                contradiction_score=contradiction,
                metadata_score=metadata,
            )

            assert (
                result.risk_score
                + result.trust_score
                == pytest.approx(100.0)
            )

    def test_custom_weights_change_risk_score(self):
        engine = RiskTrustEngine(
            injection_weight=0.70,
            poisoning_weight=0.10,
            contradiction_weight=0.10,
            metadata_weight=0.10,
        )

        result = engine.assess(
            injection_score=1.0,
            poisoning_score=0.0,
            contradiction_score=0.0,
            metadata_score=0.0,
        )

        assert result.risk_score == 70.0
        assert result.trust_score == 30.0
        assert result.classification == "QUARANTINED"

    def test_reasons_only_include_significant_signals(self):
        engine = RiskTrustEngine()

        result = engine.assess(
            injection_score=0.49,
            poisoning_score=0.49,
            contradiction_score=0.49,
            metadata_score=0.49,
        )

        assert result.reasons == []

    def test_signals_at_threshold_generate_reasons(self):
        engine = RiskTrustEngine()

        result = engine.assess(
            injection_score=0.50,
            poisoning_score=0.50,
            contradiction_score=0.50,
            metadata_score=0.50,
        )

        assert len(result.reasons) == 4

        assert "Document injection detected." in result.reasons
        assert "Potential RAG poisoning detected." in result.reasons
        assert (
            "Document contains contradictory information."
            in result.reasons
        )
        assert (
            "Suspicious document metadata detected."
            in result.reasons
        )

    def test_risk_score_never_exceeds_one_hundred(self):
        engine = RiskTrustEngine()

        result = engine.assess(
            injection_score=999,
            poisoning_score=999,
            contradiction_score=999,
            metadata_score=999,
        )

        assert result.risk_score <= 100.0
        assert result.trust_score >= 0.0

    def test_risk_score_never_goes_below_zero(self):
        engine = RiskTrustEngine()

        result = engine.assess(
            injection_score=-999,
            poisoning_score=-999,
            contradiction_score=-999,
            metadata_score=-999,
        )

        assert result.risk_score >= 0.0
        assert result.trust_score <= 100.0
        assert result.classification == "TRUSTED"