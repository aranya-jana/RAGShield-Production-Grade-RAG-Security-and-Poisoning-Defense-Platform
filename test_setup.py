#!/usr/bin/env python3

"""
Test script to verify the RAG setup using our modular components.

This file is intentionally safe to import during pytest collection.
Command-line parsing and setup verification only execute when the file
is run directly.
"""

import argparse
import os
import sys


def main() -> int:
    """Run the RAGShield setup verification."""

    parser = argparse.ArgumentParser(
        description="Test RAG system setup"
    )

    parser.add_argument(
        "--no-local",
        action="store_true",
        help="Skip local model checks (for remote inference)",
    )

    args = parser.parse_args()

    # Add src directory to path for imports.
    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(__file__),
            "src",
        ),
    )

    from config import Config

    from utils import (
        get_device,
        check_local_model_exists,
        create_chromadb_client,
        test_chromadb_operations,
        check_api_keys_file,
        get_model_cache_info,
        test_rag_components,
        test_embedding_model,
    )

    # Initialize configuration.
    # This handles environment loading, logging setup, etc.
    print("🔧 Initializing configuration...")

    if args.no_local:
        print(
            "🔧 Running in --no-local mode "
            "(skipping local model checks)"
        )

    config = Config()

    print("✅ Configuration loaded successfully")

    try:
        # Test core dependencies.
        print("✅ Core dependencies imported successfully")

        # Detect device using our utility function.
        device = get_device()

        print(f"✅ Device detected: {device}")

        # Check whether the embedding model exists locally.
        # Embeddings are always local.
        model_exists = check_local_model_exists(
            config.embedding_model,
            os.environ.get(
                "SENTENCE_TRANSFORMERS_HOME",
                "./models/embedding",
            ),
        )

        if model_exists:
            print(
                "✅ Local embedding model found: "
                f"{config.embedding_model}"
            )
        else:
            print(
                "⚠️ Local embedding model not found, "
                "may need to download: "
                f"{config.embedding_model}"
            )

        # Test embedding model.
        success, result = test_embedding_model(config)

        if success:
            print(
                "✅ Embedding model working "
                f"(dimension: {result})"
            )
        else:
            print(
                "❌ Embedding model test failed: "
                f"{result}"
            )

        # Check whether local LLM exists.
        #
        # Skip this check in --no-local mode because remote
        # inference will be used.
        if not args.no_local:
            if os.path.exists(config.llama_model_path):
                print(
                    "✅ Local LLM found at "
                    f"{config.llama_model_path}"
                )
            else:
                print(
                    "⚠️ Local LLM not found at "
                    f"{config.llama_model_path} "
                    "(optional for demo)"
                )
        else:
            print(
                "⏭️ Skipping local LLM check "
                "(--no-local mode - will use remote inference)"
            )
            print(
                "📡 Ready for remote inference providers "
                "(Ollama, DeepSeek, etc.)"
            )

        # Test ChromaDB.
        print(
            "Testing ChromaDB with path: "
            f"{config.vector_db_path}"
        )

        client = create_chromadb_client(
            config.vector_db_path
        )

        success, message = test_chromadb_operations(
            client
        )

        if success:
            print(f"✅ {message}")
        else:
            print(f"⚠️ {message}")

        # Check whether the .keys file exists.
        keys_exists, keys_path = check_api_keys_file()

        if keys_exists:
            print("✅ API Keys file (.keys) found")
        else:
            print(
                "⚠️ API Keys file (.keys) not found. "
                "Create this file to store your API keys."
            )

        # Verify local model cache directories.
        cache_info = get_model_cache_info(config)

        print(
            "✅ Embedding cache: "
            f"{cache_info['embedding_cache']}"
        )

        print(
            "✅ Transformers cache: "
            f"{cache_info['transformers_cache']}"
        )

        print(
            "✅ LLM directory: "
            f"{cache_info['llm_directory']}"
        )

        # Print configuration.
        print("\n📋 Configuration Summary:")

        config.print_config()

        # Test RAG system component integration.
        print(
            "\n🧪 Testing RAG System components..."
        )

        results, messages = test_rag_components(
            config,
            device,
            args.no_local,
        )

        for message in messages:
            print(message)

        print("\n🎉 Setup verification successful!")

        if args.no_local:
            print(
                "📡 Ready for RAG demonstration "
                "with remote inference!"
            )
            print(
                "💡 Suggested providers: "
                "Ollama, DeepSeek"
            )
        else:
            print(
                "🏴‍☠️ Ready for RAG poisoning demonstration!"
            )

        return 0

    except Exception as exc:
        print(
            f"❌ Setup verification failed: {exc}"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())