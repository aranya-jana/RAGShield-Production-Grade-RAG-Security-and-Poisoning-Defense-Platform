"""
LLM factory for creating different types of language models.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory class for creating different types of language models."""

    @staticmethod
    def create_llm(
        provider: Optional[str],
        config,
        device: Optional[str] = None
    ):
        """
        Create an LLM instance based on the provider.

        If no provider is supplied, Ollama is used by default.
        """

        # Default to Ollama for this project.
        provider = (provider or "ollama").strip().lower()

        if provider == "ollama":
            return LLMFactory._create_ollama_llm(config)

        elif provider == "openai-compat":
            return LLMFactory._create_openai_compat_llm(config)

        elif provider == "deepseek":
            return LLMFactory._create_deepseek_llm(config)

        elif provider == "llamacpp":
            return LLMFactory._create_llamacpp_llm(config, device)

        else:
            raise ValueError(
                f"Unsupported LLM provider: '{provider}'. "
                "Supported providers: ollama, openai-compat, "
                "deepseek, llamacpp."
            )

    # ------------------------------------------------------------------
    # Ollama
    # ------------------------------------------------------------------

    @staticmethod
    def _create_ollama_llm(config):
        """Create an Ollama LLM instance."""

        print(
            f"Using Ollama model: {config.ollama_model} "
            f"at {config.ollama_base_url}"
        )

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.ollama_model,
            openai_api_base=f"{config.ollama_base_url}/v1",
            openai_api_key="dummy-key",
            temperature=0,
            max_tokens=128,
            extra_body={
                "max_tokens": 128
            }
        )

    # ------------------------------------------------------------------
    # OpenAI-compatible endpoint
    # ------------------------------------------------------------------

    @staticmethod
    def _create_openai_compat_llm(config):
        """
        Create an LLM against an OpenAI-compatible endpoint.

        Supports llama-server and LM Studio.
        """

        print(
            f"Using OpenAI-compatible endpoint: "
            f"{config.openai_compat_base_url} "
            f"(model: {config.openai_compat_model})"
        )

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.openai_compat_model,
            openai_api_base=(
                f"{config.openai_compat_base_url}/v1"
            ),
            openai_api_key="dummy-key",
            temperature=0,
            max_tokens=128,
            extra_body={
                "max_tokens": 128
            }
        )

    # ------------------------------------------------------------------
    # DeepSeek
    # ------------------------------------------------------------------

    @staticmethod
    def _create_deepseek_llm(config):
        """Create DeepSeek LLM instance."""

        deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")

        if not deepseek_api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY is required for DeepSeek models. "
                "Please add it to your .keys file."
            )

        print(
            f"Using DeepSeek model: {config.deepseek_model}"
        )

        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.deepseek_model,
            openai_api_base="https://api.deepseek.com/v1",
            openai_api_key=deepseek_api_key,
            temperature=0,
            max_tokens=128
        )

    # ------------------------------------------------------------------
    # LlamaCpp
    # ------------------------------------------------------------------

    @staticmethod
    def _create_llamacpp_llm(config, device):
        """Create LlamaCpp LLM instance."""

        print(
            f"Using LlamaCpp model: {config.llama_model_path}"
        )

        from langchain_community.llms import LlamaCpp

        llama_kwargs = {
            "model_path": config.llama_model_path,
            "n_ctx": 4096,
            "n_threads": 4,
            "verbose": False
        }

        if device == "cuda":
            llama_kwargs["n_gpu_layers"] = 32

        elif device == "mps":
            llama_kwargs["n_gpu_layers"] = 1
            llama_kwargs["use_mlock"] = False

        return LlamaCpp(**llama_kwargs)


# ----------------------------------------------------------------------
# Module-level LLM wrapper
# ----------------------------------------------------------------------

def create_llm(config, provider=None, device=None):
    """
    Create an LLM using the supplied configuration.

    Supports:

        create_llm(config)

    If provider is not explicitly supplied, LLM_PROVIDER is read
    from the environment.

    If LLM_PROVIDER is not set, Ollama is used by default.
    """

    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "ollama")

    return LLMFactory.create_llm(
        provider=provider,
        config=config,
        device=device
    )


# ----------------------------------------------------------------------
# Embeddings
# ----------------------------------------------------------------------

def create_embeddings(config):
    """
    Create the embedding model used by the RAG system.
    """

    from langchain_huggingface import HuggingFaceEmbeddings

    print(
        f"Using embedding model: {config.embedding_model}"
    )

    return HuggingFaceEmbeddings(
        model_name=config.embedding_model,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )