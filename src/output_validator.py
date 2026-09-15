"""
RAGShield Output Validation & Evidence Verification
====================================================

Deterministic validation of generated RAG answers against retrieved evidence.

Security goals:
- Detect unsupported claims.
- Detect contradictions.
- Avoid accepting claims merely because they share generic words.
- Preserve evidence source/document identifiers.
- Produce safe audit telemetry without exposing complete claim text.

This is intentionally a lightweight lexical validation layer rather than an
LLM-as-judge system. It is designed to be deterministic, local, predictable,
and testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class Evidence:
    """Normalized evidence supplied by the RAG retrieval layer."""

    text: str
    source: str = ""
    document_id: str = ""


@dataclass(frozen=True)
class ClaimValidation:
    """Validation result for one extracted answer claim."""

    claim: str
    supported: bool
    contradicted: bool
    evidence_sources: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass(frozen=True)
class OutputValidationResult:
    """Complete validation result for a generated answer."""

    answer: str
    claims: List[ClaimValidation]
    is_valid: bool
    has_unsupported_claims: bool
    has_contradictions: bool
    requires_review: bool
    score: float
    reasons: List[str] = field(default_factory=list)

    @property
    def claim_count(self) -> int:
        return len(self.claims)

    @property
    def supported_claim_count(self) -> int:
        return sum(
            1
            for claim in self.claims
            if claim.supported
        )

    @property
    def unsupported_claim_count(self) -> int:
        return sum(
            1
            for claim in self.claims
            if not claim.supported
        )

    @property
    def contradicted_claim_count(self) -> int:
        return sum(
            1
            for claim in self.claims
            if claim.contradicted
        )


class OutputValidator:
    """
    Deterministic validation of generated answers against retrieved evidence.

    The validator deliberately favors false negatives over unsafe acceptance:
    when a factual claim cannot be matched to evidence with sufficient
    confidence, it is treated as unsupported.

    This is not a full natural-language entailment system. It is a security
    boundary that should be supplemented with stronger semantic verification
    in a future version.

    Important behavior:
    - Generic lexical overlap alone is not sufficient when technical values
      conflict.
    - Obvious conversational responses are not treated as factual claims.
    - Contradictory claims are separately identified.
    """

    DEFAULT_MIN_TOKEN_LENGTH = 3
    DEFAULT_SUPPORT_THRESHOLD = 0.55
    DEFAULT_CONTRADICTION_THRESHOLD = 0.55

    STOPWORDS = frozenset(
        {
            "about",
            "after",
            "again",
            "against",
            "also",
            "because",
            "before",
            "being",
            "between",
            "could",
            "does",
            "from",
            "have",
            "into",
            "more",
            "most",
            "other",
            "over",
            "same",
            "some",
            "such",
            "than",
            "that",
            "their",
            "there",
            "these",
            "they",
            "this",
            "those",
            "through",
            "under",
            "using",
            "were",
            "which",
            "while",
            "with",
            "would",
            "your",
        }
    )

    NON_FACTUAL_PATTERNS = (
        re.compile(
            r"^\s*(?:i|we)\s+can\s+help\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:sure|okay|ok|certainly|absolutely)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"^\s*(?:how\s+can\s+i|what\s+can\s+i)\s+help\b",
            re.IGNORECASE,
        ),
    )

    # Technical/value-bearing terms where disagreement is meaningful.
    #
    # Example:
    #   Evidence: SHA-256 hashing
    #   Answer:   AES-256 encryption
    #
    # These sentences share generic terms such as "verify", "document", and
    # "integrity", but the actual security mechanism differs.
    TECHNICAL_VALUE_PATTERN = re.compile(
        r"\b("
        r"sha[- ]?\d+"
        r"|aes[- ]?\d+"
        r"|md5"
        r"|sha1"
        r"|sha256"
        r"|sha512"
        r"|rsa[- ]?\d*"
        r"|ecdsa"
        r"|ed25519"
        r"|bcrypt"
        r"|argon2"
        r"|pbkdf2"
        r"|chacha20"
        r"|des"
        r"|3des"
        r"|tls\s*\d+(?:\.\d+)?"
        r"|http\s*/?\s*\d(?:\.\d+)?"
        r"|python\s*\d+(?:\.\d+)?"
        r"|node(?:\.js)?\s*\d+"
        r"|postgres(?:ql)?\s*\d+"
        r"|mysql\s*\d+"
        r"|redis\s*\d+"
        r"|kafka\s*\d+"
        r")\b",
        re.IGNORECASE,
    )

    CONTRADICTION_PATTERNS = (
        re.compile(
            r"\b(?:not|never|no longer|does not|do not|did not|"
            r"isn't|aren't|wasn't|weren't|cannot|can't|won't)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:false|incorrect|wrong|impossible|denied|"
            r"prohibited|forbidden|unsupported)\b",
            re.IGNORECASE,
        ),
    )

    def __init__(
        self,
        support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
        contradiction_threshold: float = DEFAULT_CONTRADICTION_THRESHOLD,
        min_token_length: int = DEFAULT_MIN_TOKEN_LENGTH,
    ) -> None:
        if not 0.0 <= support_threshold <= 1.0:
            raise ValueError(
                "support_threshold must be between 0 and 1."
            )

        if not 0.0 <= contradiction_threshold <= 1.0:
            raise ValueError(
                "contradiction_threshold must be between 0 and 1."
            )

        if min_token_length < 1:
            raise ValueError(
                "min_token_length must be at least 1."
            )

        self.support_threshold = support_threshold
        self.contradiction_threshold = contradiction_threshold
        self.min_token_length = min_token_length

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(
            r"\s+",
            " ",
            str(text).strip().lower(),
        )

    def _tokens(self, text: str) -> List[str]:
        normalized = self._normalize(text)

        tokens = re.findall(
            r"[a-z0-9][a-z0-9_-]*",
            normalized,
        )

        return [
            token
            for token in tokens
            if len(token) >= self.min_token_length
            and token not in self.STOPWORDS
        ]

    def _technical_values(self, text: str) -> List[str]:
        """
        Extract technical/value-bearing terms from text.

        Values are normalized so variants such as "SHA-256" and "SHA 256"
        can be compared consistently.
        """
        values = self.TECHNICAL_VALUE_PATTERN.findall(
            self._normalize(text)
        )

        return [
            re.sub(
                r"\s+",
                "",
                value.lower(),
            )
            for value in values
        ]

    def _has_conflicting_technical_values(
        self,
        claim: str,
        evidence: str,
    ) -> bool:
        """
        Detect disagreement between technical value-bearing terms.

        If both claim and evidence contain technical values but none of the
        values agree, lexical overlap must not be treated as support.
        """
        claim_values = set(
            self._technical_values(claim)
        )
        evidence_values = set(
            self._technical_values(evidence)
        )

        if not claim_values or not evidence_values:
            return False

        return not claim_values.intersection(
            evidence_values
        )

    def _is_non_factual(self, text: str) -> bool:
        """
        Return True for obvious conversational/non-factual output.

        These statements do not require retrieved evidence merely to be
        considered valid output.
        """
        normalized = self._normalize(text)

        return any(
            pattern.search(normalized)
            for pattern in self.NON_FACTUAL_PATTERNS
        )

    # ------------------------------------------------------------------
    # Claim extraction
    # ------------------------------------------------------------------

    def extract_claims(self, answer: str) -> List[str]:
        """
        Split an answer into simple factual claim candidates.

        This intentionally uses sentence-level segmentation rather than an
        LLM. Empty fragments, obvious headings, and obvious conversational
        responses are ignored.
        """
        if not isinstance(answer, str):
            raise TypeError(
                "answer must be a string."
            )

        fragments = re.split(
            r"(?<=[.!?])\s+|\n+",
            answer.strip(),
        )

        claims: List[str] = []

        for fragment in fragments:
            claim = fragment.strip()

            if not claim:
                continue

            if self._is_non_factual(claim):
                continue

            claim = re.sub(
                r"^\s*(?:[-*•]|\d+[.)])\s*",
                "",
                claim,
            ).strip()

            if not claim:
                continue

            tokens = self._tokens(claim)

            if len(tokens) < 2:
                continue

            claims.append(claim)

        return claims

    # ------------------------------------------------------------------
    # Evidence normalization
    # ------------------------------------------------------------------

    def _normalize_evidence(
        self,
        evidence: Optional[Iterable[Any]],
    ) -> List[Evidence]:
        if evidence is None:
            return []

        normalized: List[Evidence] = []

        for item in evidence:
            if isinstance(item, Evidence):
                if item.text.strip():
                    normalized.append(item)

                continue

            if isinstance(item, str):
                if item.strip():
                    normalized.append(
                        Evidence(
                            text=item
                        )
                    )

                continue

            text = getattr(
                item,
                "page_content",
                None,
            )

            if text is None:
                text = getattr(
                    item,
                    "text",
                    None,
                )

            if text is None:
                continue

            metadata = getattr(
                item,
                "metadata",
                {},
            ) or {}

            source = str(
                metadata.get(
                    "source",
                    "",
                )
                or metadata.get(
                    "file_name",
                    "",
                )
                or metadata.get(
                    "filename",
                    "",
                )
            )

            document_id = str(
                metadata.get(
                    "ragshield_document_id",
                    "",
                )
            )

            normalized.append(
                Evidence(
                    text=str(text),
                    source=source,
                    document_id=document_id,
                )
            )

        return normalized

    # ------------------------------------------------------------------
    # Similarity / evidence matching
    # ------------------------------------------------------------------

    def _token_overlap(
        self,
        claim: str,
        evidence: str,
    ) -> float:
        claim_tokens = set(
            self._tokens(claim)
        )

        evidence_tokens = set(
            self._tokens(evidence)
        )

        if not claim_tokens:
            return 0.0

        overlap = claim_tokens.intersection(
            evidence_tokens
        )

        return len(overlap) / len(claim_tokens)

    def _phrase_match(
        self,
        claim: str,
        evidence: str,
    ) -> bool:
        claim_normalized = self._normalize(
            claim
        )

        evidence_normalized = self._normalize(
            evidence
        )

        return (
            len(claim_normalized) >= 12
            and claim_normalized
            in evidence_normalized
        )

    def _support_score(
        self,
        claim: str,
        evidence: str,
    ) -> float:
        """
        Calculate deterministic lexical support.

        Technical-value conflicts are rejected before calculating ordinary
        lexical overlap to avoid false support.
        """
        if self._has_conflicting_technical_values(
            claim,
            evidence,
        ):
            return 0.0

        overlap = self._token_overlap(
            claim,
            evidence,
        )

        if self._phrase_match(
            claim,
            evidence,
        ):
            return max(
                overlap,
                1.0,
            )

        return overlap

    # ------------------------------------------------------------------
    # Contradiction detection
    # ------------------------------------------------------------------

    def _negative_form(self, text: str) -> str:
        normalized = self._normalize(
            text
        )

        normalized = re.sub(
            r"\b(is|are|was|were|has|have|does|do|did|can|will)\b",
            "",
            normalized,
        )

        normalized = re.sub(
            r"\b(?:not|never|no longer|does not|do not|did not|"
            r"isn't|aren't|wasn't|weren't|cannot|can't|won't)\b",
            "",
            normalized,
        )

        return re.sub(
            r"\s+",
            " ",
            normalized,
        ).strip()

    def _is_potential_contradiction(
        self,
        claim: str,
        evidence: str,
    ) -> bool:
        claim_normalized = self._normalize(
            claim
        )

        evidence_normalized = self._normalize(
            evidence
        )

        claim_has_negative = any(
            pattern.search(claim_normalized)
            for pattern in self.CONTRADICTION_PATTERNS[:1]
        )

        evidence_has_negative = any(
            pattern.search(evidence_normalized)
            for pattern in self.CONTRADICTION_PATTERNS[:1]
        )

        if claim_has_negative == evidence_has_negative:
            return False

        claim_base = self._negative_form(
            claim_normalized
        )

        evidence_base = self._negative_form(
            evidence_normalized
        )

        if not claim_base or not evidence_base:
            return False

        return (
            self._token_overlap(
                claim_base,
                evidence_base,
            )
            >= self.contradiction_threshold
        )

    # ------------------------------------------------------------------
    # Claim validation
    # ------------------------------------------------------------------

    def validate_claim(
        self,
        claim: str,
        evidence: Sequence[Evidence],
    ) -> ClaimValidation:
        best_score = 0.0
        best_sources: List[str] = []
        contradicted = False

        for item in evidence:
            score = self._support_score(
                claim,
                item.text,
            )

            if score > best_score:
                best_score = score

                best_sources = [
                    value
                    for value in (
                        item.source,
                        item.document_id,
                    )
                    if value
                ]

            if (
                score
                >= self.contradiction_threshold
                and self._is_potential_contradiction(
                    claim,
                    item.text,
                )
            ):
                contradicted = True

        supported = (
            best_score
            >= self.support_threshold
            and not contradicted
        )

        if contradicted:
            reason = (
                "Claim appears to contradict retrieved evidence."
            )
        elif supported:
            reason = (
                "Claim has sufficient lexical support "
                "from retrieved evidence."
            )
        else:
            reason = (
                "Claim could not be sufficiently supported "
                "by retrieved evidence."
            )

        return ClaimValidation(
            claim=claim,
            supported=supported,
            contradicted=contradicted,
            evidence_sources=best_sources,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Complete validation
    # ------------------------------------------------------------------

    def validate(
        self,
        answer: str,
        evidence: Optional[Iterable[Any]],
    ) -> OutputValidationResult:
        if not isinstance(answer, str):
            raise TypeError(
                "answer must be a string."
            )

        normalized_evidence = (
            self._normalize_evidence(
                evidence
            )
        )

        claims = self.extract_claims(
            answer
        )

        validations = [
            self.validate_claim(
                claim,
                normalized_evidence,
            )
            for claim in claims
        ]

        has_unsupported = any(
            not claim.supported
            for claim in validations
        )

        has_contradictions = any(
            claim.contradicted
            for claim in validations
        )

        if not validations:
            score = 1.0
        else:
            score = (
                sum(
                    1.0
                    if claim.supported
                    else 0.0
                    for claim in validations
                )
                / len(validations)
            )

        reasons: List[str] = []

        if (
            not normalized_evidence
            and claims
        ):
            reasons.append(
                "No retrieved evidence was available "
                "to validate factual claims."
            )

        if has_unsupported:
            reasons.append(
                "One or more generated claims lack sufficient "
                "retrieved evidence."
            )

        if has_contradictions:
            reasons.append(
                "One or more generated claims appear to "
                "contradict retrieved evidence."
            )

        is_valid = (
            bool(claims)
            and not has_unsupported
            and not has_contradictions
        )

        # An answer with no factual claim candidates, such as a basic
        # conversational response, is valid and does not require evidence.
        if not claims:
            is_valid = True

        return OutputValidationResult(
            answer=answer,
            claims=validations,
            is_valid=is_valid,
            has_unsupported_claims=has_unsupported,
            has_contradictions=has_contradictions,
            requires_review=(
                has_unsupported
                or has_contradictions
                or (
                    not normalized_evidence
                    and bool(claims)
                )
            ),
            score=score,
            reasons=reasons,
        )

    # ------------------------------------------------------------------
    # Safe telemetry
    # ------------------------------------------------------------------

    def safe_audit(
        self,
        result: OutputValidationResult,
    ) -> Dict[str, Any]:
        """
        Return telemetry safe for audit logs and API responses.

        Complete claim text is intentionally excluded.
        """
        claims: List[Dict[str, Any]] = []

        for claim in result.claims:
            claims.append(
                {
                    "supported": bool(
                        claim.supported
                    ),
                    "contradicted": bool(
                        claim.contradicted
                    ),
                    "evidence_sources": list(
                        claim.evidence_sources
                    ),
                    "reason": claim.reason,
                    "claim_length": len(
                        claim.claim
                    ),
                }
            )

        return {
            "is_valid": bool(
                result.is_valid
            ),
            "has_unsupported_claims": bool(
                result.has_unsupported_claims
            ),
            "has_contradictions": bool(
                result.has_contradictions
            ),
            "requires_review": bool(
                result.requires_review
            ),
            "score": float(
                result.score
            ),
            "claim_count": int(
                result.claim_count
            ),
            "supported_claim_count": int(
                result.supported_claim_count
            ),
            "unsupported_claim_count": int(
                result.unsupported_claim_count
            ),
            "contradicted_claim_count": int(
                result.contradicted_claim_count
            ),
            "reasons": list(
                result.reasons
            ),
            "claims": claims,
        }


def validate_output(
    answer: str,
    evidence: Optional[Iterable[Any]],
) -> OutputValidationResult:
    """Convenience function for one-shot output validation."""
    return OutputValidator().validate(
        answer,
        evidence,
    )