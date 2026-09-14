#!/usr/bin/env python3

"""
RAG Poisoning Demonstration

Demonstrates:

1. A benign RAG system
2. A poisoned RAG system
3. Prompt/instruction poisoning detection
4. Factual contradiction detection
5. Protection/blocking of poisoned documents
6. Custom poisoning payloads
7. Reconstructed prompt inspection
"""

import argparse
import logging
import sys
import time

from src.config import Config
from src.llm_factory import create_embeddings, create_llm
from src.rag_system import RAGSystem
from src.rag_poisoning_corpus import (
    create_benign_corpus,
    create_poisoned_document,
)
from src.poison_detector import PoisonDetector
from src.contradiction_detector import ContradictionDetector


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# DISPLAY HELPERS
# ----------------------------------------------------------------------

def print_separator():
    """Print a visual separator."""
    print("\n" + "=" * 80 + "\n")


def print_subseparator():
    """Print a smaller separator."""
    print("-" * 80)


# ----------------------------------------------------------------------
# DOCUMENT ANALYSIS
# ----------------------------------------------------------------------

def analyze_documents():
    """
    Analyze the benign and poisoned corpus using both detectors.

    PoisonDetector:
        Detects instruction/prompt manipulation.

    ContradictionDetector:
        Detects factual contradiction against trusted documents.
    """

    print("Document Poison Detection")
    print_separator()

    # --------------------------------------------------------------
    # Create corpus
    # --------------------------------------------------------------

    benign_docs = create_benign_corpus()
    poisoned_doc = create_poisoned_document()

    documents = benign_docs + [poisoned_doc]

    print(
        f"Total documents: {len(documents)}"
    )

    # --------------------------------------------------------------
    # Poison detector
    # --------------------------------------------------------------

    poison_detector = PoisonDetector()

    # --------------------------------------------------------------
    # Contradiction detector
    #
    # It needs the same embedding model used by the RAG system.
    # --------------------------------------------------------------

    contradiction_detector = None

    try:

        config = Config()

        embeddings = create_embeddings(
            config
        )

        contradiction_detector = (
            ContradictionDetector(
                embeddings=embeddings,
                threshold=0.80,
            )
        )

    except Exception as exc:

        logger.warning(
            "Could not initialize ContradictionDetector: "
            f"{type(exc).__name__}: {exc}"
        )

        print(
            "WARNING: ContradictionDetector could not "
            "be initialized."
        )

    # --------------------------------------------------------------
    # Analyze every document
    # --------------------------------------------------------------

    for index, document in enumerate(
        documents,
        start=1,
    ):

        source = document.metadata.get(
            "source",
            "unknown",
        )

        print(
            f"\nDocument {index}"
        )

        print(
            f"Source: {source}"
        )

        print_subseparator()

        # ----------------------------------------------------------
        # PoisonDetector
        # ----------------------------------------------------------

        poison_result = poison_detector.analyze(
            document.page_content
        )

        print(
            f"Poison score: "
            f"{poison_result.score:.2f}"
        )

        print(
            f"Poisoned: "
            f"{poison_result.is_poisoned}"
        )

        if poison_result.reasons:

            print(
                "Poison reasons:"
            )

            for reason in poison_result.reasons:

                print(
                    f"  - {reason}"
                )

        else:

            print(
                "Poison reasons: None"
            )

        # ----------------------------------------------------------
        # ContradictionDetector
        #
        # Only the candidate poisoned document needs to be checked
        # against the trusted benign corpus for this POC.
        # ----------------------------------------------------------

        if (
            contradiction_detector is not None
            and document is poisoned_doc
        ):

            contradiction_result = (
                contradiction_detector.analyze(
                    candidate_text=document.page_content,
                    trusted_documents=benign_docs,
                )
            )

            print(
                f"\nContradiction score: "
                f"{contradiction_result.score:.2f}"
            )

            print(
                f"Contradictory: "
                f"{contradiction_result.is_contradictory}"
            )

            if contradiction_result.reasons:

                print(
                    "Contradiction reasons:"
                )

                for reason in (
                    contradiction_result.reasons
                ):

                    print(
                        f"  - {reason}"
                    )

            else:

                print(
                    "Contradiction reasons: None"
                )

    print_separator()


# ----------------------------------------------------------------------
# RAG CREATION
# ----------------------------------------------------------------------

def build_rag(
    config,
    include_poison=False,
    payload=None,
):
    """
    Create and populate a RAGSystem.
    """

    print(
        f"Setting up RAG system "
        f"(poison={include_poison})..."
    )

    # --------------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------------

    embeddings = create_embeddings(
        config
    )

    # --------------------------------------------------------------
    # LLM
    # --------------------------------------------------------------

    llm = create_llm(
        config
    )

    # --------------------------------------------------------------
    # RAG system
    # --------------------------------------------------------------

    rag = RAGSystem(
        config=config,
        embeddings=embeddings,
        llm=llm,
    )

    # --------------------------------------------------------------
    # Populate vector database
    # --------------------------------------------------------------

    rag.setup_vector_database(
        include_poison=include_poison,
        payload=payload,
    )

    # --------------------------------------------------------------
    # Refresh retrieval chain
    # --------------------------------------------------------------

    rag.refresh_chain()

    return rag


# ----------------------------------------------------------------------
# QUERY
# ----------------------------------------------------------------------

def run_query(
    rag,
    query,
):
    """
    Run one protected RAG query and display results.
    """

    print(
        f"\nQuery: {query}"
    )

    print("-" * 80)

    start = time.time()

    result = rag.query(
        query
    )

    elapsed = time.time() - start

    # --------------------------------------------------------------
    # Answer
    # --------------------------------------------------------------

    answer = result.get(
        "result",
        "",
    )

    print(
        "\nAnswer:"
    )

    print(
        answer
    )

    # --------------------------------------------------------------
    # Safe documents
    # --------------------------------------------------------------

    source_documents = result.get(
        "source_documents",
        [],
    )

    print(
        "\nRetrieved documents:"
    )

    if not source_documents:

        print(
            "  No source documents returned."
        )

    else:

        for index, document in enumerate(
            source_documents,
            start=1,
        ):

            source = document.metadata.get(
                "source",
                "unknown",
            )

            doc_type = document.metadata.get(
                "type",
                "unknown",
            )

            print(
                f"  {index}. "
                f"{source} "
                f"(type={doc_type})"
            )

    # --------------------------------------------------------------
    # Blocked documents
    # --------------------------------------------------------------

    blocked_documents = result.get(
        "blocked_documents",
        [],
    )

    print(
        "\nBlocked documents:"
    )

    if not blocked_documents:

        print(
            "  None"
        )

    else:

        for index, document in enumerate(
            blocked_documents,
            start=1,
        ):

            source = document.metadata.get(
                "source",
                "unknown",
            )

            doc_type = document.metadata.get(
                "type",
                "unknown",
            )

            print(
                f"  {index}. "
                f"{source} "
                f"(type={doc_type})"
            )

    # --------------------------------------------------------------
    # Timing
    # --------------------------------------------------------------

    print(
        f"\nQuery time: {elapsed:.2f}s"
    )

    return result


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="RAG poisoning demonstration"
    )

    # --------------------------------------------------------------
    # Analyze only
    # --------------------------------------------------------------

    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help=(
            "Only run poison and contradiction detection "
            "against the corpus"
        ),
    )

    # --------------------------------------------------------------
    # Poison
    # --------------------------------------------------------------

    parser.add_argument(
        "--poison",
        action="store_true",
        help=(
            "Include the poisoned document in the "
            "RAG database"
        ),
    )

    # --------------------------------------------------------------
    # Query
    # --------------------------------------------------------------

    parser.add_argument(
        "--query",
        type=str,
        default="What is cloud computing?",
        help="Question to ask the RAG system",
    )

    # --------------------------------------------------------------
    # Show prompt
    # --------------------------------------------------------------

    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help=(
            "Display the prompt constructed from "
            "retrieved documents"
        ),
    )

    # --------------------------------------------------------------
    # Custom payload
    # --------------------------------------------------------------

    parser.add_argument(
        "--payload",
        type=str,
        default=None,
        help=(
            "Custom document payload for the "
            "poisoning demonstration"
        ),
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Logging
    # --------------------------------------------------------------

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # --------------------------------------------------------------
    # Header
    # --------------------------------------------------------------

    print_separator()

    print(
        "RAG POISONING DEMONSTRATION"
    )

    print_separator()

    # ==============================================================
    # STEP 1
    # ==============================================================

    analyze_documents()

    if args.analyze_only:

        print(
            "Analysis-only mode complete."
        )

        return 0

    # ==============================================================
    # STEP 2
    # ==============================================================

    print(
        "Loading configuration..."
    )

    try:

        config = Config()

    except Exception as exc:

        print(
            f"ERROR: Could not load configuration: "
            f"{type(exc).__name__}: {exc}"
        )

        logger.exception(
            "Configuration loading failed"
        )

        return 1

    print(
        "Configuration loaded."
    )

    # ==============================================================
    # STEP 3
    # ==============================================================

    try:

        rag = build_rag(
            config=config,
            include_poison=args.poison,
            payload=args.payload,
        )

    except Exception as exc:

        print(
            "\nERROR: Failed to initialize RAG system."
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        logger.exception(
            "RAG initialization failed"
        )

        return 1

    print(
        "\nRAG system initialized successfully."
    )

    # ==============================================================
    # STEP 4
    # ==============================================================

    try:

        result = run_query(
            rag,
            args.query,
        )

    except Exception as exc:

        print(
            "\nERROR: Query failed."
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        logger.exception(
            "RAG query failed"
        )

        return 1

    # ==============================================================
    # STEP 5
    # ==============================================================

    if args.show_prompt:

        source_documents = result.get(
            "source_documents",
            [],
        )

        if source_documents:

            print_separator()

            print(
                "RECONSTRUCTED LLM PROMPT"
            )

            print_separator()

            try:

                prompt = rag.format_prompt(
                    args.query,
                    source_documents,
                )

                print(
                    prompt
                )

            except Exception as exc:

                print(
                    "Could not reconstruct prompt: "
                    f"{type(exc).__name__}: {exc}"
                )

        else:

            print(
                "\nNo safe source documents available; "
                "cannot reconstruct prompt."
            )

    # ==============================================================
    # COMPLETE
    # ==============================================================

    print_separator()

    print(
        "Demo complete."
    )

    print_separator()

    return 0


# ----------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(
        main()
    )