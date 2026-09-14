"""
Factual contradiction detection for RAG poisoning.

This detector compares a candidate document against trusted documents
using the existing embedding model.

It is intentionally separate from PoisonDetector:

- PoisonDetector -> prompt/instruction manipulation
- ContradictionDetector -> semantic/factual conflict
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ContradictionResult:
    score: float
    is_contradictory: bool
    reasons: List[str]


class ContradictionDetector:
    """
    Detect semantic/factual conflict between a candidate document
    and trusted reference documents.

    Embedding similarity is used only as a screening signal.
    Similarity alone does NOT prove contradiction.
    """

    def __init__(
        self,
        embeddings=None,
        threshold: float = 0.80,
    ):
        self.embeddings = embeddings
        self.threshold = threshold

    def _get_embedding(self, text: str):
        """Generate an embedding for text."""
        if self.embeddings is None:
            raise ValueError(
                "Embeddings are required for ContradictionDetector."
            )

        return self.embeddings.embed_query(text)

    @staticmethod
    def _cosine_similarity(a, b) -> float:
        """Calculate cosine similarity."""
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)

        denominator = np.linalg.norm(a) * np.linalg.norm(b)

        if denominator == 0:
            return 0.0

        return float(np.dot(a, b) / denominator)

    def compare(
        self,
        candidate_text: str,
        trusted_text: str,
    ) -> float:
        """
        Return semantic similarity between candidate and trusted text.

        High similarity means the documents discuss similar subject matter.
        It does NOT by itself mean they agree.
        """

        if not candidate_text or not trusted_text:
            return 0.0

        candidate_embedding = self._get_embedding(candidate_text)
        trusted_embedding = self._get_embedding(trusted_text)

        return self._cosine_similarity(
            candidate_embedding,
            trusted_embedding,
        )

    def analyze(
        self,
        candidate_text: str,
        trusted_documents: Optional[list] = None,
    ) -> ContradictionResult:
        """
        Analyze a candidate document against trusted documents.
        """

        if not candidate_text:
            return ContradictionResult(
                score=0.0,
                is_contradictory=False,
                reasons=[],
            )

        if not trusted_documents:
            return ContradictionResult(
                score=0.0,
                is_contradictory=False,
                reasons=[],
            )

        candidate_lower = candidate_text.lower()

        best_similarity = 0.0
        best_document = None

        # --------------------------------------------------------------
        # Find the most semantically similar trusted document
        # --------------------------------------------------------------

        for document in trusted_documents:
            trusted_text = getattr(
                document,
                "page_content",
                "",
            )

            if not trusted_text:
                continue

            similarity = self.compare(
                candidate_text,
                trusted_text,
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_document = document

        reasons = []
        score = 0.0

        # --------------------------------------------------------------
        # Conservative contradiction heuristics
        # --------------------------------------------------------------

        contradiction_pairs = [

            (
                [
                    "does not use the internet",
                    "doesn't use the internet",
                    "not use the internet",
                ],
                [
                    "over the internet",
                    "internet",
                ],
                (
                    "Claims cloud computing does not use the internet "
                    "while trusted context describes internet-based access."
                ),
            ),

            (
                [
                    "cannot scale",
                    "cannot scale dynamically",
                    "does not scale",
                    "doesn't scale",
                ],
                [
                    "scalability",
                    "scale dynamically",
                    "scalable",
                ],
                (
                    "Claims cloud computing cannot scale while trusted "
                    "context describes scalability."
                ),
            ),

            (
                [
                    "no cost benefits",
                    "provides no cost benefits",
                    "not cost efficient",
                ],
                [
                    "cost efficiency",
                    "cost efficient",
                    "cost benefits",
                ],
                (
                    "Claims there are no cost benefits while trusted "
                    "context describes cost efficiency."
                ),
            ),
        ]

        trusted_texts = [
            getattr(
                document,
                "page_content",
                "",
            ).lower()
            for document in trusted_documents
        ]

        combined_trusted = "\n".join(trusted_texts)

        for (
            candidate_phrases,
            trusted_phrases,
            reason,
        ) in contradiction_pairs:

            candidate_has_claim = any(
                phrase in candidate_lower
                for phrase in candidate_phrases
            )

            trusted_has_opposite = any(
                phrase in combined_trusted
                for phrase in trusted_phrases
            )

            if candidate_has_claim and trusted_has_opposite:

                score += 0.35
                reasons.append(reason)

        # --------------------------------------------------------------
        # Semantic similarity strengthens an explicit contradiction
        # --------------------------------------------------------------

        if reasons and best_similarity >= self.threshold:

            score += 0.25

            if best_document is not None:

                metadata = getattr(
                    best_document,
                    "metadata",
                    {},
                )

                source = metadata.get(
                    "source",
                    "trusted document",
                )

                reasons.append(
                    f"Semantic conflict with trusted context: {source}"
                )

        # --------------------------------------------------------------
        # Clamp score
        # --------------------------------------------------------------

        score = min(
            1.0,
            round(score, 3),
        )

        return ContradictionResult(
            score=score,
            is_contradictory=score >= 0.50,
            reasons=reasons,
        )