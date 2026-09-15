from datetime import datetime
from types import SimpleNamespace

from src.document_provenance import (
    CONTENT_HASH_KEY,
    DOCUMENT_ID_KEY,
    INGESTED_AT_KEY,
    METADATA_HASH_KEY,
    PROVENANCE_VERSION_KEY,
    SOURCE_KEY,
    DocumentProvenanceManager,
)


def make_document(
    content="This is a trusted document.",
    metadata=None,
):
    return SimpleNamespace(
        page_content=content,
        metadata=dict(metadata or {}),
    )


def test_content_hash_is_deterministic():
    first = DocumentProvenanceManager.hash_content("hello")
    second = DocumentProvenanceManager.hash_content("hello")

    assert first == second
    assert (
        first
        == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_metadata_hash_is_order_independent():
    first = DocumentProvenanceManager.hash_metadata(
        {
            "source": "internal.pdf",
            "author": "security-team",
            "version": "1",
        }
    )

    second = DocumentProvenanceManager.hash_metadata(
        {
            "version": "1",
            "author": "security-team",
            "source": "internal.pdf",
        }
    )

    assert first == second


def test_provenance_creation_contains_required_fields():
    provenance = DocumentProvenanceManager.create_provenance(
        content="Trusted content",
        metadata={"source": "internal.pdf"},
    )

    metadata = provenance.to_metadata()

    assert provenance.document_id
    assert provenance.source == "internal.pdf"
    assert provenance.ingested_at
    assert provenance.content_sha256
    assert provenance.metadata_sha256
    assert provenance.provenance_version == "1.0"

    assert DOCUMENT_ID_KEY in metadata
    assert SOURCE_KEY in metadata
    assert INGESTED_AT_KEY in metadata
    assert CONTENT_HASH_KEY in metadata
    assert METADATA_HASH_KEY in metadata
    assert PROVENANCE_VERSION_KEY in metadata


def test_document_id_is_stable_for_same_source_and_content():
    first = DocumentProvenanceManager.create_provenance(
        content="Same content",
        metadata={"source": "document.pdf"},
    )

    second = DocumentProvenanceManager.create_provenance(
        content="Same content",
        metadata={"source": "document.pdf"},
    )

    assert first.document_id == second.document_id


def test_document_id_changes_when_content_changes():
    first = DocumentProvenanceManager.create_provenance(
        content="Original content",
        metadata={"source": "document.pdf"},
    )

    second = DocumentProvenanceManager.create_provenance(
        content="Modified content",
        metadata={"source": "document.pdf"},
    )

    assert first.document_id != second.document_id


def test_enrich_document_preserves_original_metadata():
    document = make_document(
        metadata={
            "source": "trusted.pdf",
            "document_type": "policy",
        }
    )

    result = DocumentProvenanceManager.enrich_document(document)

    assert result is document
    assert document.metadata["source"] == "trusted.pdf"
    assert document.metadata["document_type"] == "policy"

    assert DOCUMENT_ID_KEY in document.metadata
    assert CONTENT_HASH_KEY in document.metadata
    assert METADATA_HASH_KEY in document.metadata


def test_clean_document_passes_integrity_verification():
    document = make_document(
        content="Trusted content",
        metadata={"source": "trusted.pdf"},
    )

    DocumentProvenanceManager.enrich_document(document)

    result = DocumentProvenanceManager.verify_document(document)

    assert result.is_valid is True
    assert result.tampered is False
    assert result.content_hash_valid is True
    assert result.metadata_hash_valid is True
    assert result.reasons == ()


def test_content_tampering_is_detected():
    document = make_document(
        content="Original trusted content",
        metadata={"source": "trusted.pdf"},
    )

    DocumentProvenanceManager.enrich_document(document)

    document.page_content = "Attacker modified this content."

    result = DocumentProvenanceManager.verify_document(document)

    assert result.is_valid is False
    assert result.tampered is True
    assert result.content_hash_valid is False
    assert result.metadata_hash_valid is True
    assert any("content SHA-256" in reason for reason in result.reasons)


def test_metadata_tampering_is_detected():
    document = make_document(
        content="Original trusted content",
        metadata={"source": "trusted.pdf", "classification": "internal"},
    )

    DocumentProvenanceManager.enrich_document(document)

    document.metadata["classification"] = "public"

    result = DocumentProvenanceManager.verify_document(document)

    assert result.is_valid is False
    assert result.tampered is True
    assert result.content_hash_valid is True
    assert result.metadata_hash_valid is False
    assert any("metadata SHA-256" in reason for reason in result.reasons)


def test_missing_content_hash_is_detected():
    document = make_document(
        content="Trusted content",
        metadata={"source": "trusted.pdf"},
    )

    DocumentProvenanceManager.enrich_document(document)
    del document.metadata[CONTENT_HASH_KEY]

    result = DocumentProvenanceManager.verify_document(document)

    assert result.is_valid is False
    assert result.content_hash_valid is False
    assert any("missing its stored content" in reason for reason in result.reasons)


def test_missing_metadata_hash_is_detected():
    document = make_document(
        content="Trusted content",
        metadata={"source": "trusted.pdf"},
    )

    DocumentProvenanceManager.enrich_document(document)
    del document.metadata[METADATA_HASH_KEY]

    result = DocumentProvenanceManager.verify_document(document)

    assert result.is_valid is False
    assert result.metadata_hash_valid is False
    assert any("missing its stored metadata" in reason for reason in result.reasons)


def test_provenance_fields_do_not_change_metadata_hash():
    original_metadata = {
        "source": "trusted.pdf",
        "document_type": "policy",
    }

    original_hash = DocumentProvenanceManager.hash_metadata(
        original_metadata
    )

    enriched_metadata = dict(original_metadata)
    provenance = DocumentProvenanceManager.create_provenance(
        content="Trusted content",
        metadata=original_metadata,
    )
    enriched_metadata.update(provenance.to_metadata())

    enriched_hash = DocumentProvenanceManager.hash_metadata(
        enriched_metadata
    )

    assert original_hash == enriched_hash


def test_ingestion_timestamp_is_valid_iso_datetime():
    provenance = DocumentProvenanceManager.create_provenance(
        content="Trusted content",
        metadata={"source": "trusted.pdf"},
    )

    parsed = datetime.fromisoformat(provenance.ingested_at)

    assert parsed.tzinfo is not None


def test_tampered_provenance_hash_is_reported():
    document = make_document(
        content="Trusted content",
        metadata={"source": "trusted.pdf"},
    )

    DocumentProvenanceManager.enrich_document(document)

    document.metadata[CONTENT_HASH_KEY] = "0" * 64

    result = DocumentProvenanceManager.verify_document(document)

    assert result.is_valid is False
    assert result.content_hash_valid is False