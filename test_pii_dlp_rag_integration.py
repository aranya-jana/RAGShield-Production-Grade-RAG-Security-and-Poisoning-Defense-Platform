from types import SimpleNamespace

from langchain_core.documents import Document

from src.pii_detector import PIIDetector
from src.rag_system import RAGSystem


def make_rag_for_dlp():
    rag = RAGSystem.__new__(RAGSystem)
    rag.pii_detector = PIIDetector()
    return rag


def test_document_dlp_detects_ordinary_pii_without_blocking():
    rag = make_rag_for_dlp()
    doc = Document(
        page_content="Contact security@example.com for support.",
        metadata={"source": "contact.md"},
    )

    result = rag._analyze_document_dlp(doc)

    assert result.has_pii is True
    assert "email" in result.categories
    assert rag._dlp_contains_high_risk_secret(result) is False


def test_document_dlp_detects_high_risk_secret():
    rag = make_rag_for_dlp()
    doc = Document(
        page_content="api_key=super-secret-value-12345",
        metadata={"source": "config.txt"},
    )

    result = rag._analyze_document_dlp(doc)

    assert result.has_pii is True
    assert "generic_secret" in result.categories
    assert rag._dlp_contains_high_risk_secret(result) is True


def test_document_dlp_scans_metadata():
    rag = make_rag_for_dlp()
    doc = Document(
        page_content="Internal configuration document.",
        metadata={
            "source": "config.txt",
            "owner_email": "admin@example.com",
        },
    )

    result = rag._analyze_document_dlp(doc)

    assert result.has_pii is True
    assert "email" in result.categories


def test_dlp_audit_metadata_masks_raw_values():
    rag = make_rag_for_dlp()
    doc = Document(
        page_content="Contact attacker@example.com.",
        metadata={"source": "attack.md"},
    )

    result = rag._analyze_document_dlp(doc)
    metadata = rag._document_dlp_metadata(result)

    assert metadata["has_pii"] is True
    assert "email" in metadata["categories"]
    assert metadata["findings"]
    assert metadata["findings"][0]["value_masked"] != "attacker@example.com"
    assert "attacker@example.com" not in str(metadata)


def test_dlp_metadata_excludes_ragshield_provenance_fields():
    rag = make_rag_for_dlp()
    doc = Document(
        page_content="No sensitive data here.",
        metadata={
            "source": "safe.md",
            "ragshield_source": "safe.md",
            "ragshield_content_sha256": "not-sensitive",
        },
    )

    result = rag._analyze_document_dlp(doc)

    assert result.has_pii is False


def test_dlp_clean_document_is_safe():
    rag = make_rag_for_dlp()
    doc = Document(
        page_content="RAGShield protects retrieved context.",
        metadata={"source": "safe.md"},
    )

    result = rag._analyze_document_dlp(doc)

    assert result.has_pii is False
    assert result.score == 0.0
    assert result.findings == ()


def test_dlp_high_risk_categories_are_blocking():
    rag = make_rag_for_dlp()

    for text, category in [
        ("SSN: 123-45-6789", "ssn"),
        ("AWS: AKIAIOSFODNN7EXAMPLE", "aws_access_key"),
        (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "secret-material\n"
            "-----END RSA PRIVATE KEY-----",
            "private_key",
        ),
    ]:
        result = rag._analyze_document_dlp(
            Document(
                page_content=text,
                metadata={"source": "sensitive.txt"},
            )
        )

        assert category in result.categories
        assert rag._dlp_contains_high_risk_secret(result) is True
