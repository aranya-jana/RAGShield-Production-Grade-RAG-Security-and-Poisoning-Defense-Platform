import pytest

from src.rag_system import RAGSystem


@pytest.fixture
def rag():
    rag = RAGSystem.__new__(RAGSystem)
    rag.pii_detector = None
    rag.output_dlp_redact_categories = {
        "credit_card",
        "ssn",
        "aws_access_key",
        "github_token",
        "private_key",
        "jwt",
        "generic_secret",
    }
    return rag


def test_output_dlp_allows_clean_response(rag):
    result = rag._analyze_output_dlp("RAGShield is working correctly.")
    assert result.action == "ALLOWED"
    assert result.redacted_text == "RAGShield is working correctly."
    assert not result.detection.has_pii


def test_output_dlp_detects_ordinary_pii_without_redacting(rag):
    result = rag._analyze_output_dlp("Contact alice@example.com for details.")
    assert result.action == "ALLOWED"
    assert result.redacted_text == "Contact alice@example.com for details."
    assert result.detection.has_pii
    assert "email" in result.detection.categories


def test_output_dlp_redacts_credit_card(rag):
    result = rag._analyze_output_dlp("The card is 4111 1111 1111 1111.")
    assert result.action == "REDACTED"
    assert "4111 1111 1111 1111" not in result.redacted_text
    assert "[REDACTED:credit_card]" in result.redacted_text


def test_output_dlp_redacts_aws_key(rag):
    result = rag._analyze_output_dlp("Credential: AKIAIOSFODNN7EXAMPLE")
    assert result.action == "REDACTED"
    assert "AKIAIOSFODNN7EXAMPLE" not in result.redacted_text


def test_output_dlp_redacts_private_key(rag):
    result = rag._analyze_output_dlp(
        "-----BEGIN PRIVATE KEY----- secret material -----END PRIVATE KEY-----"
    )
    assert result.action == "REDACTED"
    assert "BEGIN PRIVATE KEY" not in result.redacted_text


def test_output_dlp_metadata_masks_raw_values(rag):
    result = rag._analyze_output_dlp(
        "Email alice@example.com and card 4111 1111 1111 1111."
    )
    metadata = rag._output_dlp_metadata(result)
    assert metadata["action"] == "REDACTED"
    serialized = str(metadata["findings"])
    assert "alice@example.com" not in serialized
    assert "4111 1111 1111 1111" not in serialized


def test_output_dlp_none_is_safe(rag):
    metadata = rag._output_dlp_metadata(None)
    assert metadata["action"] == "ALLOWED"
    assert metadata["findings"] == []
