"""
RAGShield Document Provenance & Integrity.

Provides deterministic document provenance metadata and SHA-256
integrity verification for RAG documents.

Security properties:

- deterministic content hashing
- deterministic metadata hashing
- stable document identifiers
- ingestion timestamp tracking
- provenance metadata
- retrieval-time integrity verification

Important limitation:

SHA-256 provides integrity detection, not cryptographic authenticity.
An attacker who can modify both a document and its stored hash can
recompute the hash. Strong authenticity would require a secret-backed
HMAC or digital signature layer.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Dict, Optional, Tuple


PROVENANCE_VERSION = "1.0"

DOCUMENT_ID_KEY = "ragshield_document_id"
SOURCE_KEY = "ragshield_source"
INGESTED_AT_KEY = "ragshield_ingested_at"
CONTENT_HASH_KEY = "ragshield_content_sha256"
METADATA_HASH_KEY = "ragshield_metadata_sha256"
PROVENANCE_VERSION_KEY = "ragshield_provenance_version"

_PROVENANCE_KEYS = {
    DOCUMENT_ID_KEY,
    SOURCE_KEY,
    INGESTED_AT_KEY,
    CONTENT_HASH_KEY,
    METADATA_HASH_KEY,
    PROVENANCE_VERSION_KEY,
}


@dataclass(frozen=True)
class DocumentProvenance:
    """Immutable provenance and integrity information."""

    document_id: str
    source: str
    ingested_at: str
    content_sha256: str
    metadata_sha256: str
    provenance_version: str = PROVENANCE_VERSION

    def to_metadata(self) -> Dict[str, str]:
        """Return provenance fields suitable for LangChain metadata."""

        return {
            DOCUMENT_ID_KEY: self.document_id,
            SOURCE_KEY: self.source,
            INGESTED_AT_KEY: self.ingested_at,
            CONTENT_HASH_KEY: self.content_sha256,
            METADATA_HASH_KEY: self.metadata_sha256,
            PROVENANCE_VERSION_KEY: self.provenance_version,
        }


@dataclass(frozen=True)
class IntegrityVerification:
    """Result returned when document integrity is verified."""

    is_valid: bool
    content_hash_valid: bool
    metadata_hash_valid: bool
    expected_content_sha256: str
    actual_content_sha256: str
    expected_metadata_sha256: str
    actual_metadata_sha256: str
    reasons: Tuple[str, ...] = ()

    @property
    def tampered(self) -> bool:
        """Return True when any integrity check failed."""

        return not self.is_valid


class DocumentProvenanceManager:
    """
    Create and verify RAGShield document provenance.

    The manager does not mutate caller-owned metadata dictionaries
    directly. It creates a normalized copy before enrichment.
    """

    @staticmethod
    def _normalize_content(content: Any) -> str:
        """Normalize document content into a deterministic string."""

        if content is None:
            return ""

        return str(content)

    @staticmethod
    def _canonicalize_metadata(
        metadata: Optional[Dict[str, Any]],
    ) -> str:
        """
        Convert security-relevant metadata into deterministic JSON.

        RAGShield-generated provenance fields are excluded because they
        contain the hashes being calculated and would otherwise create
        circular hashing.
        """

        metadata = dict(metadata or {})

        filtered = {
            str(key): value
            for key, value in metadata.items()
            if str(key) not in _PROVENANCE_KEYS
        }

        try:
            return json.dumps(
                filtered,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            )
        except (TypeError, ValueError):
            return json.dumps(
                {
                    str(key): str(value)
                    for key, value in filtered.items()
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )

    @classmethod
    def hash_content(cls, content: Any) -> str:
        """Return SHA-256 for document content."""

        normalized = cls._normalize_content(content)

        return hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

    @classmethod
    def hash_metadata(
        cls,
        metadata: Optional[Dict[str, Any]],
    ) -> str:
        """Return SHA-256 for security-relevant document metadata."""

        canonical = cls._canonicalize_metadata(
            metadata
        )

        return hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()

    @classmethod
    def create_document_id(
        cls,
        source: Any,
        content_sha256: str,
    ) -> str:
        """
        Create a stable document identifier.

        The identifier is derived from source and content hash, making
        identical source/content pairs deterministic.
        """

        source_text = str(
            source if source is not None else "unknown"
        ).strip()

        identity_material = (
            f"{source_text}\n{content_sha256}"
        )

        return hashlib.sha256(
            identity_material.encode("utf-8")
        ).hexdigest()

    @classmethod
    def create_provenance(
        cls,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
        ingested_at: Optional[str] = None,
    ) -> DocumentProvenance:
        """Create provenance information for a document."""

        metadata_copy = dict(metadata or {})

        resolved_source = (
            source
            if source is not None
            else metadata_copy.get("source", "unknown")
        )

        resolved_source = str(
            resolved_source
            if resolved_source is not None
            else "unknown"
        )

        content_sha256 = cls.hash_content(
            content
        )

        metadata_sha256 = cls.hash_metadata(
            metadata_copy
        )

        document_id = cls.create_document_id(
            resolved_source,
            content_sha256,
        )

        resolved_ingested_at = (
            ingested_at
            if ingested_at is not None
            else datetime.now(
                timezone.utc
            ).isoformat()
        )

        return DocumentProvenance(
            document_id=document_id,
            source=resolved_source,
            ingested_at=resolved_ingested_at,
            content_sha256=content_sha256,
            metadata_sha256=metadata_sha256,
        )

    @classmethod
    def enrich_document(cls, document):
        """
        Add RAGShield provenance metadata to a LangChain Document.

        Returns the same document object for convenient pipeline
        integration.
        """

        metadata = dict(
            getattr(document, "metadata", {}) or {}
        )

        provenance = cls.create_provenance(
            content=getattr(
                document,
                "page_content",
                "",
            ),
            metadata=metadata,
        )

        metadata.update(
            provenance.to_metadata()
        )

        document.metadata = metadata

        return document

    @classmethod
    def verify_document(
        cls,
        document,
    ) -> IntegrityVerification:
        """
        Verify a document's stored provenance against current content
        and metadata.
        """

        metadata = dict(
            getattr(document, "metadata", {}) or {}
        )

        expected_content = str(
            metadata.get(
                CONTENT_HASH_KEY,
                "",
            )
        )

        expected_metadata = str(
            metadata.get(
                METADATA_HASH_KEY,
                "",
            )
        )

        actual_content = cls.hash_content(
            getattr(
                document,
                "page_content",
                "",
            )
        )

        actual_metadata = cls.hash_metadata(
            metadata
        )

        content_valid = (
            bool(expected_content)
            and expected_content == actual_content
        )

        metadata_valid = (
            bool(expected_metadata)
            and expected_metadata == actual_metadata
        )

        reasons = []

        if not expected_content:
            reasons.append(
                "Document is missing its stored content SHA-256 hash."
            )
        elif not content_valid:
            reasons.append(
                "Document content SHA-256 does not match the stored integrity hash."
            )

        if not expected_metadata:
            reasons.append(
                "Document is missing its stored metadata SHA-256 hash."
            )
        elif not metadata_valid:
            reasons.append(
                "Document metadata SHA-256 does not match the stored integrity hash."
            )

        return IntegrityVerification(
            is_valid=(
                content_valid
                and metadata_valid
            ),
            content_hash_valid=content_valid,
            metadata_hash_valid=metadata_valid,
            expected_content_sha256=expected_content,
            actual_content_sha256=actual_content,
            expected_metadata_sha256=expected_metadata,
            actual_metadata_sha256=actual_metadata,
            reasons=tuple(reasons),
        )


__all__ = [
    "PROVENANCE_VERSION",
    "DOCUMENT_ID_KEY",
    "SOURCE_KEY",
    "INGESTED_AT_KEY",
    "CONTENT_HASH_KEY",
    "METADATA_HASH_KEY",
    "PROVENANCE_VERSION_KEY",
    "DocumentProvenance",
    "IntegrityVerification",
    "DocumentProvenanceManager",
]