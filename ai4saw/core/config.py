"""Environment variable loading and validated config object."""

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # Provider
    provider: Literal["ollama", "openrouter", "huggingface"] = "ollama"

    # Models
    default_model: str = "mistral"
    embedding_model: str = "nomic-embed-text"

    # API keys / endpoints
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    hf_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # ChromaDB
    chroma_persist_dir: Path = Path("./data/chroma")
    chroma_collection: str = "ai4saw"

    # Extraction
    max_tokens_per_run: int = 500_000
    extraction_retry_limit: int = 2

    # Retrieval
    retrieval_top_k: int = 8
    rerank_top_n: int = 3

    # Logging
    log_level: str = "INFO"
    log_file: Path = Path("./logs/ai4saw.log")

    # Paths
    prompts_dir: Path = Path("./prompts")
    output_dir: Path = Path("./output")
    corpus_dir: Path = Path("./corpus")

    @field_validator("chroma_persist_dir", "log_file", "output_dir", mode="before")
    @classmethod
    def _make_parents(cls, v: str | Path) -> Path:
        p = Path(v)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Singleton — import this everywhere instead of re-parsing env
settings = Settings()
