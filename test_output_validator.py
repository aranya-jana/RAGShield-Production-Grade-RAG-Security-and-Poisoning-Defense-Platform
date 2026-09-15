from src.output_validator import (
    Evidence,
    OutputValidator,
    validate_output,
)


def test_supported_claim_is_valid():
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
    assert result.claim_count == 1
    assert result.supported_claim_count == 1
    assert result.unsupported_claim_count == 0
    assert result.score == 1.0


def test_unsupported_claim_is_detected():
    validator = OutputValidator()

    evidence = [
        Evidence(
            text=(
                "RAGShield uses SHA-256 hashing to verify "
                "document integrity."
            ),
            source="security-policy.md",
        )
    ]

    result = validator.validate(
        "RAGShield uses AES-256 encryption to verify document integrity.",
        evidence,
    )

    assert result.is_valid is False
    assert result.has_unsupported_claims is True
    assert result.has_contradictions is False
    assert result.unsupported_claim_count == 1
    assert result.requires_review is True
    assert result.score < 1.0


def test_multiple_supported_claims_are_valid():
    validator = OutputValidator()

    evidence = [
        Evidence(
            text=(
                "RAGShield detects prompt injection attacks "
                "before the LLM is invoked."
            ),
            source="prompt-security.md",
        ),
        Evidence(
            text=(
                "RAGShield records security events in an "
                "audit log."
            ),
            source="audit-policy.md",
        ),
    ]

    result = validator.validate(
        (
            "RAGShield detects prompt injection attacks before "
            "the LLM is invoked. RAGShield records security "
            "events in an audit log."
        ),
        evidence,
    )

    assert result.is_valid is True
    assert result.claim_count == 2
    assert result.supported_claim_count == 2
    assert result.unsupported_claim_count == 0
    assert result.score == 1.0


def test_mixed_supported_and_unsupported_claims_require_review():
    validator = OutputValidator()

    evidence = [
        Evidence(
            text=(
                "RAGShield detects prompt injection attacks "
                "before the LLM is invoked."
            ),
            source="prompt-security.md",
        )
    ]

    result = validator.validate(
        (
            "RAGShield detects prompt injection attacks before "
            "the LLM is invoked. RAGShield automatically deletes "
            "all malicious documents."
        ),
        evidence,
    )

    assert result.is_valid is False
    assert result.has_unsupported_claims is True
    assert result.requires_review is True
    assert result.supported_claim_count == 1
    assert result.unsupported_claim_count == 1
    assert result.score == 0.5


def test_contradictory_claim_is_detected():
    validator = OutputValidator()

    evidence = [
        Evidence(
            text=(
                "The production API does not permit anonymous "
                "queries."
            ),
            source="api-security.md",
        )
    ]

    result = validator.validate(
        "The production API permits anonymous queries.",
        evidence,
    )

    assert result.is_valid is False
    assert result.has_contradictions is True
    assert result.requires_review is True
    assert result.contradicted_claim_count >= 1


def test_negative_claim_can_be_supported_by_negative_evidence():
    validator = OutputValidator()

    evidence = [
        Evidence(
            text=(
                "The production API does not permit anonymous "
                "queries."
            ),
            source="api-security.md",
        )
    ]

    result = validator.validate(
        "The production API does not permit anonymous queries.",
        evidence,
    )

    assert result.is_valid is True
    assert result.has_contradictions is False
    assert result.has_unsupported_claims is False


def test_evidence_source_is_returned():
    validator = OutputValidator()

    evidence = [
        Evidence(
            text=(
                "Document provenance is verified using "
                "SHA-256 hashes."
            ),
            source="provenance.md",
            document_id="doc-provenance-001",
        )
    ]

    result = validator.validate(
        "Document provenance is verified using SHA-256 hashes.",
        evidence,
    )

    claim = result.claims[0]

    assert claim.supported is True
    assert claim.contradicted is False
    assert "provenance.md" in claim.evidence_sources
    assert "doc-provenance-001" in claim.evidence_sources


def test_langchain_document_evidence_is_supported():
    validator = OutputValidator()

    class FakeDocument:
        page_content = (
            "RAGShield uses provenance hashes to verify "
            "document integrity."
        )
        metadata = {
            "source": "provenance.md",
            "ragshield_document_id": "doc-123",
        }

    result = validator.validate(
        "RAGShield uses provenance hashes to verify document integrity.",
        [FakeDocument()],
    )

    assert result.is_valid is True
    assert result.claims[0].supported is True
    assert "provenance.md" in result.claims[0].evidence_sources
    assert "doc-123" in result.claims[0].evidence_sources


def test_string_evidence_is_supported():
    validator = OutputValidator()

    result = validator.validate(
        "RAGShield protects generated output with DLP.",
        [
            "RAGShield protects generated output with DLP."
        ],
    )

    assert result.is_valid is True
    assert result.supported_claim_count == 1


def test_no_evidence_requires_review():
    validator = OutputValidator()

    result = validator.validate(
        "RAGShield provides enterprise-grade security.",
        [],
    )

    assert result.is_valid is False
    assert result.has_unsupported_claims is True
    assert result.requires_review is True
    assert result.score == 0.0
    assert any(
        "No retrieved evidence" in reason
        for reason in result.reasons
    )


def test_empty_or_non_claim_output_is_allowed():
    validator = OutputValidator()

    result = validator.validate(
        "Okay.",
        [],
    )

    assert result.is_valid is True
    assert result.claim_count == 0
    assert result.has_unsupported_claims is False
    assert result.has_contradictions is False


def test_claim_extraction_handles_markdown_bullets():
    validator = OutputValidator()

    claims = validator.extract_claims(
        (
            "- RAGShield detects prompt injection attacks.\n"
            "- RAGShield records security events."
        )
    )

    assert len(claims) == 2
    assert "RAGShield detects prompt injection attacks." in claims
    assert "RAGShield records security events." in claims


def test_safe_audit_does_not_expose_full_claim_text():
    validator = OutputValidator()

    sensitive_claim = (
        "The administrator email is alice@example.com."
    )

    evidence = [
        Evidence(
            text=sensitive_claim,
            source="internal.md",
        )
    ]

    result = validator.validate(
        sensitive_claim,
        evidence,
    )

    audit = validator.safe_audit(result)

    audit_text = str(audit)

    assert sensitive_claim not in audit_text
    assert "alice@example.com" not in audit_text

    assert audit["claim_count"] == 1
    assert audit["supported_claim_count"] == 1
    assert audit["unsupported_claim_count"] == 0


def test_safe_audit_contains_validation_metrics():
    validator = OutputValidator()

    result = validator.validate(
        "RAGShield uses SHA-256 hashing.",
        [
            Evidence(
                text="RAGShield uses SHA-256 hashing.",
                source="security.md",
            )
        ],
    )

    audit = validator.safe_audit(result)

    assert audit["is_valid"] is True
    assert audit["has_unsupported_claims"] is False
    assert audit["has_contradictions"] is False
    assert audit["requires_review"] is False
    assert audit["score"] == 1.0
    assert audit["claim_count"] == 1
    assert audit["supported_claim_count"] == 1
    assert audit["unsupported_claim_count"] == 0
    assert audit["contradicted_claim_count"] == 0


def test_convenience_function_validate_output():
    result = validate_output(
        "RAGShield uses SHA-256 hashing.",
        [
            "RAGShield uses SHA-256 hashing."
        ],
    )

    assert result.is_valid is True
    assert result.supported_claim_count == 1


def test_invalid_threshold_is_rejected():
    try:
        OutputValidator(
            support_threshold=1.5
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "between 0 and 1" in str(exc)


def test_invalid_min_token_length_is_rejected():
    try:
        OutputValidator(
            min_token_length=0
        )
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "at least 1" in str(exc)


def test_non_string_answer_is_rejected():
    validator = OutputValidator()

    try:
        validator.validate(
            None,
            [],
        )
        assert False, "Expected TypeError"
    except TypeError as exc:
        assert "answer" in str(exc).lower()