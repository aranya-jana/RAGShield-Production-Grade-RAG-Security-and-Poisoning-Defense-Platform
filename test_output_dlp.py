from src.output_dlp import OutputDLP, protect_output


def test_clean_output_is_allowed_unchanged():
    dlp = OutputDLP()

    result = dlp.analyze(
        "RAGShield found three relevant documents."
    )

    assert result.blocked is False
    assert result.redacted is False
    assert result.protected_text == (
        "RAGShield found three relevant documents."
    )
    assert result.has_findings is False
    assert result.finding_count == 0


def test_email_output_is_redacted():
    dlp = OutputDLP()

    result = dlp.analyze(
        "Contact the administrator at alice@example.com."
    )

    assert result.blocked is False
    assert result.redacted is True
    assert result.protected_text != result.original_text
    assert "alice@example.com" not in result.protected_text
    assert "[REDACTED:email]" in result.protected_text
    assert "email" in result.categories


def test_phone_output_is_redacted():
    dlp = OutputDLP()

    result = dlp.analyze(
        "Call the support team at 415-555-2671."
    )

    assert result.blocked is False
    assert result.redacted is True
    assert "415-555-2671" not in result.protected_text
    assert "[REDACTED:phone]" in result.protected_text
    assert "phone" in result.categories


def test_credit_card_output_is_blocked():
    dlp = OutputDLP()

    result = dlp.analyze(
        "The customer's card number is 4111 1111 1111 1111."
    )

    assert result.blocked is True
    assert result.redacted is True
    assert result.protected_text != result.original_text
    assert "4111 1111 1111 1111" not in result.protected_text
    assert "credit_card" in result.high_risk_categories


def test_ssn_output_is_blocked():
    dlp = OutputDLP()

    result = dlp.analyze(
        "The employee SSN is 123-45-6789."
    )

    assert result.blocked is True
    assert "123-45-6789" not in result.protected_text
    assert "ssn" in result.high_risk_categories


def test_aws_key_output_is_blocked():
    dlp = OutputDLP()

    result = dlp.analyze(
        "Use AWS key "
        "AKIAIOSFODNN7EXAMPLE "
        "for the deployment."
    )

    assert result.blocked is True
    assert "AKIAIOSFODNN7EXAMPLE" not in result.protected_text
    assert "aws_access_key" in result.high_risk_categories


def test_github_token_output_is_blocked():
    dlp = OutputDLP()

    result = dlp.analyze(
        "The GitHub token is "
        "ghp_1234567890abcdefghijklmnopqrstuvwxyz."
    )

    assert result.blocked is True
    assert "ghp_1234567890abcdefghijklmnopqrstuvwxyz" not in (
        result.protected_text
    )
    assert "github_token" in result.high_risk_categories


def test_private_key_output_is_blocked():
    dlp = OutputDLP()

    result = dlp.analyze(
        "-----BEGIN PRIVATE KEY-----\n"
        "secret-material\n"
        "-----END PRIVATE KEY-----"
    )

    assert result.blocked is True
    assert "private_key" in result.high_risk_categories


def test_jwt_output_is_blocked():
    dlp = OutputDLP()

    result = dlp.analyze(
        "Bearer "
        "eyJhbGciOiJIUzI1NiJ9."
        "eyJzdWIiOiIxMjM0NTYifQ."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )

    assert result.blocked is True
    assert "jwt" in result.high_risk_categories


def test_mixed_pii_and_secret_is_blocked():
    dlp = OutputDLP()

    result = dlp.analyze(
        "Contact alice@example.com and use "
        "AWS key AKIAIOSFODNN7EXAMPLE."
    )

    assert result.blocked is True
    assert result.redacted is True
    assert "alice@example.com" not in result.protected_text
    assert "AKIAIOSFODNN7EXAMPLE" not in result.protected_text
    assert "email" in result.categories
    assert "aws_access_key" in result.high_risk_categories


def test_safe_audit_never_contains_raw_sensitive_values():
    dlp = OutputDLP()

    result = dlp.analyze(
        "Contact alice@example.com or use "
        "AWS key AKIAIOSFODNN7EXAMPLE."
    )

    audit = dlp.safe_audit(result)

    audit_text = str(audit)

    assert "alice@example.com" not in audit_text
    assert "AKIAIOSFODNN7EXAMPLE" not in audit_text

    assert audit["blocked"] is True
    assert audit["redacted"] is True
    assert audit["finding_count"] >= 2
    assert "email" in audit["categories"]
    assert "aws_access_key" in audit["high_risk_categories"]


def test_convenience_function_protect_output():
    result = protect_output(
        "Send the report to bob@example.com."
    )

    assert result.blocked is False
    assert result.redacted is True
    assert "bob@example.com" not in result.protected_text


def test_non_string_output_is_rejected():
    dlp = OutputDLP()

    try:
        dlp.analyze(None)
        assert False, "Expected TypeError"
    except TypeError as exc:
        assert "string" in str(exc).lower()


def test_custom_blocked_category_policy():
    dlp = OutputDLP(
        blocked_categories={"email"}
    )

    result = dlp.analyze(
        "Contact alice@example.com."
    )

    assert result.blocked is True
    assert result.redacted is True
    assert "email" in result.high_risk_categories


def test_output_dlp_does_not_modify_clean_text():
    dlp = OutputDLP()

    text = (
        "The security analysis completed successfully. "
        "No threats were detected."
    )

    result = dlp.analyze(text)

    assert result.protected_text == text
    assert result.original_text == text