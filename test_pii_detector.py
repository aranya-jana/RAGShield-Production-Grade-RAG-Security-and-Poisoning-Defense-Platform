from src.pii_detector import PIIDetector


def test_clean_text_has_no_pii():
    detector = PIIDetector()

    result = detector.analyze(
        "RAGShield is a security-hardened retrieval system."
    )

    assert result.has_pii is False
    assert result.score == 0.0
    assert result.findings == ()
    assert result.categories == ()


def test_email_is_detected():
    detector = PIIDetector()

    result = detector.analyze(
        "Contact security@example.com for assistance."
    )

    assert result.has_pii is True
    assert "email" in result.categories
    assert result.finding_count == 1


def test_phone_number_is_detected():
    detector = PIIDetector()

    result = detector.analyze(
        "Call +1 415-555-1234 for support."
    )

    assert result.has_pii is True
    assert "phone" in result.categories


def test_credit_card_is_detected():
    detector = PIIDetector()

    result = detector.analyze(
        "Payment card: 4111 1111 1111 1111"
    )

    assert result.has_pii is True
    assert "credit_card" in result.categories
    assert result.score >= 1.0


def test_ssn_is_detected():
    detector = PIIDetector()

    result = detector.analyze(
        "SSN: 123-45-6789"
    )

    assert result.has_pii is True
    assert "ssn" in result.categories


def test_aws_access_key_is_detected():
    detector = PIIDetector()

    result = detector.analyze(
        "AWS key: AKIAIOSFODNN7EXAMPLE"
    )

    assert result.has_pii is True
    assert "aws_access_key" in result.categories


def test_github_token_is_detected():
    detector = PIIDetector()

    result = detector.analyze(
        "Token: ghp_abcdefghijklmnopqrstuvwxyz123456"
    )

    assert result.has_pii is True
    assert "github_token" in result.categories


def test_private_key_is_detected():
    detector = PIIDetector()

    result = detector.analyze(
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "secret-material\n"
        "-----END RSA PRIVATE KEY-----"
    )

    assert result.has_pii is True
    assert "private_key" in result.categories


def test_jwt_is_detected():
    detector = PIIDetector()

    result = detector.analyze(
        "Authorization: "
        "eyJhbGciOiJIUzI1NiJ9."
        "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )

    assert result.has_pii is True
    assert "jwt" in result.categories


def test_generic_secret_is_detected():
    detector = PIIDetector()

    result = detector.analyze(
        "api_key=super-secret-value-12345"
    )

    assert result.has_pii is True
    assert "generic_secret" in result.categories


def test_multiple_sensitive_categories_are_detected():
    detector = PIIDetector()

    result = detector.analyze(
        "Email: attacker@example.com "
        "and API key: abcdefghijklmnop1234"
    )

    assert result.has_pii is True
    assert "email" in result.categories
    assert "generic_secret" in result.categories
    assert result.finding_count >= 2


def test_none_input_is_safe():
    detector = PIIDetector()

    result = detector.analyze(None)

    assert result.has_pii is False
    assert result.score == 0.0
    assert result.findings == ()


def test_redaction_removes_detected_sensitive_values():
    detector = PIIDetector()

    text = "Contact attacker@example.com for assistance."

    result = detector.analyze(text)

    redacted = detector.redact_findings(
        text,
        result,
    )

    assert "attacker@example.com" not in redacted
    assert "[REDACTED:email]" in redacted


def test_audit_findings_do_not_expose_raw_values():
    detector = PIIDetector()

    text = "Contact attacker@example.com."

    result = detector.analyze(text)

    audit_findings = detector.safe_audit_findings(result)

    assert len(audit_findings) == 1
    assert audit_findings[0]["category"] == "email"
    assert audit_findings[0]["value_masked"] != "attacker@example.com"
    assert "attacker@example.com" not in str(audit_findings)


def test_threshold_controls_detection():
    detector = PIIDetector(threshold=0.90)

    result = detector.analyze(
        "Contact security@example.com."
    )

    assert result.score < 0.90
    assert result.has_pii is False
    assert "email" in result.categories