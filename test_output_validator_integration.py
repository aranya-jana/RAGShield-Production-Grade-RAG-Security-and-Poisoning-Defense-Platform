"""
RAGShield Output Validator Integration Tests
============================================

Tests the expected integration contract between the RAG system and the
deterministic OutputValidator.

These tests are intentionally written before modifying rag_system.py so the
security behavior is explicit and regression-safe.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.output_validator import Evidence, OutputValidator


def test_supported_generated_answer_is_validated_against_retrieved_evidence():
    validator = OutputValidator()

    evidence = [
        Evidence(
            text=(
                "RAGShield uses SHA-256 hashing to verify "
                "document integrity."
            ),
            source="security-policy.md",
            document_id="doc-001",
        )
    ]

    result = validator.validate(
        "RAGShield uses SHA-256 hashing to verify document integrity.",
        evidence,
    )

    assert result.is_valid is True
    assert result.has_unsupported_claims is False
    assert result.has_contradictions is False
    assert result.requires_review is False
    assert result.score == 1.0


def test_unsupported_generated_answer_requires_review():
    validator = OutputValidator()

    evidence = [
        Evidence(
            text=(
                "RAGShield uses SHA-256 hashing to verify "
                "document integrity."
            ),
            source="security-policy.md",
            document_id="doc-001",
        )
    ]

    result = validator.validate(
        "RAGShield uses AES-256 encryption to verify document integrity.",
        evidence,
    )

    assert result.is_valid is False
    assert result.has_unsupported_claims is True
    assert result.requires_review is True
    assert result.unsupported_claim_count == 1


def test_contradictory_generated_answer_is_detected():
    validator = OutputValidator()

    evidence = [
        Evidence(
            text=(
                "The production API does not permit anonymous queries."
            ),
            source="api-security.md",
            document_id="doc-002",
        )
    ]

    result = validator.validate(
        "The production API permits anonymous queries.",
        evidence,
    )

    assert result.is_valid is False
    assert result.has_contradictions is True
    assert result.requires_review is True
    assert result.contradicted_claim_count == 1


def test_supported_negative_claim_is_not_marked_as_contradiction():
    validator = OutputValidator()

    evidence = [
        Evidence(
            text=(
                "The production API does not permit anonymous queries."
            ),
            source="api-security.md",
            document_id="doc-002",
        )
    ]

    result = validator.validate(
        "The production API does not permit anonymous queries.",
        evidence,
    )

    assert result.is_valid is True
    assert result.has_contradictions is False
    assert result.has_unsupported_claims is False


def test_multiple_claims_require_each_claim_to_be_supported():
    validator = OutputValidator()

    evidence = [
        Evidence(
            text=(
                "RAGShield uses SHA-256 hashing for document integrity."
            ),
            source="security-policy.md",
            document_id="doc-001",
        ),
        Evidence(
            text=(
                "RAGShield uses PBKDF2 for password hashing."
            ),
            source="auth-policy.md",
            document_id="doc-003",
        ),
    ]

    result = validator.validate(
        (
            "RAGShield uses SHA-256 hashing for document integrity. "
            "RAGShield uses PBKDF2 for password hashing."
        ),
        evidence,
    )

    assert result.is_valid is True
    assert result.claim_count == 2
    assert result.supported_claim_count == 2
    assert result.unsupported_claim_count == 0


def test_mixed_supported_and_unsupported_claims_require_review():
    validator = OutputValidator()

    evidence = [
        Evidence(
            text=(
                "RAGShield uses SHA-256 hashing for document integrity."
            ),
            source="security-policy.md",
            document_id="doc-001",
        )
    ]

    result = validator.validate(
        (
            "RAGShield uses SHA-256 hashing for document integrity. "
            "RAGShield encrypts every document with AES-256."
        ),
        evidence,
    )

    assert result.is_valid is False
    assert result.has_unsupported_claims is True
    assert result.requires_review is True
    assert result.supported_claim_count == 1
    assert result.unsupported_claim_count == 1


def test_langchain_style_documents_are_supported():
    validator = OutputValidator()

    document = SimpleNamespace(
        page_content=(
            "RAGShield stores document integrity hashes using SHA-256."
        ),
        metadata={
            "source": "integrity.md",
            "ragshield_document_id": "doc-004",
        },
    )

    result = validator.validate(
        "RAGShield stores document integrity hashes using SHA-256.",
        [document],
    )

    assert result.is_valid is True
    assert result.claims[0].evidence_sources == [
        "integrity.md",
        "doc-004",
    ]


def test_no_evidence_with_factual_claim_requires_review():
    validator = OutputValidator()

    result = validator.validate(
        "RAGShield uses SHA-256 for document integrity.",
        [],
    )

    assert result.is_valid is False
    assert result.has_unsupported_claims is True
    assert result.requires_review is True
    assert result.score == 0.0


def test_empty_or_non_factual_output_does_not_create_false_security_failure():
    validator = OutputValidator()

    result = validator.validate(
        "I can help you with that.",
        [],
    )

    assert result.is_valid is True
    assert result.claim_count == 0
    assert result.has_unsupported_claims is False
    assert result.has_contradictions is False