"""
Extended tests for the RAG ContradictionDetector.

The current proof-of-concept detector uses semantic similarity
together with known factual contradiction patterns.

These tests verify:

1. Empty input handling.
2. Missing trusted context handling.
3. Internet contradictions.
4. Scalability contradictions.
5. Cost contradictions.
6. Multiple factual contradictions.
7. Consistent information.
8. Similarity calculations.
9. Score bounds.
"""

import pytest

from langchain_core.documents import Document

from src.contradiction_detector import ContradictionDetector


# =====================================================================
# MOCK EMBEDDINGS
# =====================================================================


class MockEmbeddings:
    """
    Deterministic embeddings used for unit testing.

    Every document receives the same vector so semantic similarity
    is deterministic and does not depend on HuggingFace models.
    """

    def embed_query(self, text):
        return [1.0, 0.0]

    def embed_documents(self, texts):
        return [
            [1.0, 0.0]
            for _ in texts
        ]


# =====================================================================
# FIXTURES
# =====================================================================


@pytest.fixture
def detector():
    return ContradictionDetector(
        embeddings=MockEmbeddings(),
        threshold=0.50,
    )


@pytest.fixture
def trusted_cloud_document():
    return Document(
        page_content="""
        Cloud computing provides on-demand access to computing
        resources over the internet. Cloud platforms provide
        scalability and cost efficiency.
        """,
        metadata={
            "source": "cloud_computing_guide.md"
        },
    )


# =====================================================================
# EMPTY / MISSING CONTEXT
# =====================================================================


def test_empty_candidate_is_clean(detector):
    result = detector.analyze(
        candidate_text="",
        trusted_documents=[],
    )

    assert result.is_contradictory is False
    assert result.score == 0.0


def test_no_trusted_documents_is_clean(detector):
    result = detector.analyze(
        candidate_text=(
            "Cloud computing does not use the internet."
        ),
        trusted_documents=[],
    )

    assert result.is_contradictory is False
    assert result.score == 0.0


# =====================================================================
# INTERNET CONTRADICTION
# =====================================================================


def test_internet_contradiction_is_detected(
    detector,
    trusted_cloud_document,
):
    candidate = """
    Cloud computing does not use the internet.
    """

    result = detector.analyze(
        candidate_text=candidate,
        trusted_documents=[
            trusted_cloud_document
        ],
    )

    assert result.is_contradictory is True
    assert result.score >= detector.threshold
    assert len(result.reasons) > 0


# =====================================================================
# SCALABILITY CONTRADICTION
# =====================================================================


def test_scalability_contradiction_is_detected(
    detector,
    trusted_cloud_document,
):
    candidate = """
    Cloud computing cannot scale dynamically.
    """

    result = detector.analyze(
        candidate_text=candidate,
        trusted_documents=[
            trusted_cloud_document
        ],
    )

    assert result.is_contradictory is True
    assert result.score >= detector.threshold


# =====================================================================
# COST CONTRADICTION
# =====================================================================


def test_cost_contradiction_is_detected(
    detector,
    trusted_cloud_document,
):
    candidate = """
    Cloud computing provides no cost benefits.
    """

    result = detector.analyze(
        candidate_text=candidate,
        trusted_documents=[
            trusted_cloud_document
        ],
    )

    assert result.is_contradictory is True
    assert result.score >= detector.threshold


# =====================================================================
# MULTIPLE CONTRADICTIONS
# =====================================================================


def test_multiple_factual_contradictions_are_detected(
    detector,
    trusted_cloud_document,
):
    candidate = """
    Cloud computing does not use the internet.
    It cannot scale dynamically.
    It provides no cost benefits.
    """

    result = detector.analyze(
        candidate_text=candidate,
        trusted_documents=[
            trusted_cloud_document
        ],
    )

    assert result.is_contradictory is True
    assert result.score >= detector.threshold

    assert len(result.reasons) >= 3


# =====================================================================
# CONSISTENT INFORMATION
# =====================================================================


def test_consistent_document_is_not_contradictory(
    detector,
    trusted_cloud_document,
):
    candidate = """
    Cloud computing provides access to computing resources
    over the internet and offers scalability and cost efficiency.
    """

    result = detector.analyze(
        candidate_text=candidate,
        trusted_documents=[
            trusted_cloud_document
        ],
    )

    assert result.is_contradictory is False


# =====================================================================
# SIMILARITY
# =====================================================================


def test_similarity_calculation(detector):
    similarity = detector._cosine_similarity(
        [1.0, 0.0],
        [1.0, 0.0],
    )

    assert similarity == pytest.approx(1.0)


def test_orthogonal_vectors_have_zero_similarity(detector):
    similarity = detector._cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    )

    assert similarity == pytest.approx(0.0)


# =====================================================================
# SCORE BOUNDS
# =====================================================================


def test_contradiction_score_is_bounded(
    detector,
    trusted_cloud_document,
):
    candidate = """
    Cloud computing does not use the internet.
    It cannot scale dynamically.
    It provides no cost benefits.
    """

    result = detector.analyze(
        candidate_text=candidate,
        trusted_documents=[
            trusted_cloud_document
        ],
    )

    assert 0.0 <= result.score <= 1.0