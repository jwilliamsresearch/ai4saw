"""Provider abstraction layer.

All LLM and embedding calls route through get_llm() and get_embedder().
No pipeline module references a provider directly.
Switching providers requires only a change to the PROVIDER env var.
"""

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from ai4saw.core.config import settings


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Return a configured chat model for the active provider."""
    match settings.provider:
        case "ollama":
            return _ollama_llm()
        case "openrouter":
            return _openrouter_llm()
        case "huggingface":
            return _huggingface_llm()
        case _:
            raise ValueError(f"Unknown provider: {settings.provider!r}")


@lru_cache(maxsize=1)
def get_embedder() -> Embeddings:
    """Return a configured embeddings model for the active provider."""
    match settings.provider:
        case "ollama":
            return _ollama_embedder()
        case "openrouter":
            # OpenRouter does not expose an embeddings API; fall back to local.
            return _ollama_embedder()
        case "huggingface":
            return _huggingface_embedder()
        case _:
            raise ValueError(f"Unknown provider: {settings.provider!r}")


# ── Private constructors ───────────────────────────────────────────────────────

def _ollama_llm() -> BaseChatModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.default_model,
        base_url=settings.ollama_base_url,
        temperature=0.0,
    )


def _ollama_embedder() -> Embeddings:
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )


def _openrouter_llm() -> BaseChatModel:
    # OpenRouter exposes an OpenAI-compatible API.
    from langchain_openai import ChatOpenAI

    if not settings.openrouter_api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. "
            "Add it to your .env file or set PROVIDER=ollama for local inference."
        )

    return ChatOpenAI(
        model=settings.default_model,
        api_key=settings.openrouter_api_key,  # type: ignore[arg-type]
        base_url=settings.openrouter_base_url,
        temperature=0.0,
    )


def _huggingface_llm() -> BaseChatModel:
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    if not settings.hf_api_key:
        raise EnvironmentError(
            "HF_API_KEY is not set. "
            "Add it to your .env file or set PROVIDER=ollama for local inference."
        )

    endpoint = HuggingFaceEndpoint(
        repo_id=settings.default_model,
        huggingfacehub_api_token=settings.hf_api_key,
        temperature=0.01,  # HF endpoint does not support 0.0
    )
    return ChatHuggingFace(llm=endpoint)


def _huggingface_embedder() -> Embeddings:
    from langchain_huggingface import HuggingFaceEndpointEmbeddings

    if not settings.hf_api_key:
        raise EnvironmentError("HF_API_KEY is not set.")

    return HuggingFaceEndpointEmbeddings(
        model=f"sentence-transformers/{settings.embedding_model}",
        huggingfacehub_api_token=settings.hf_api_key,
    )
