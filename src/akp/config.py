from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # "offline" runs the whole graph with deterministic rule-based agents:
    # no API key, used by tests and CI.
    llm_provider: Literal["anthropic", "offline"] = "offline"
    model_router: str = "claude-opus-5-5"
    model_answer: str = "claude-opus-5-5"
    model_critic: str = "claude-opus-5-5"
    effort_router: Literal["low", "medium", "high"] = "low"
    effort_answer: Literal["low", "medium", "high"] = "medium"
    effort_critic: Literal["low", "medium", "high"] = "low"

    qdrant_url: str = ":memory:"
    collection: str = "documents"
    embedding_dim: int = 1024
    chunk_size: int = 700
    chunk_overlap: int = 100
    top_k: int = 4

    max_retrieval_rounds: int = 2
    seed_corpus_dir: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
