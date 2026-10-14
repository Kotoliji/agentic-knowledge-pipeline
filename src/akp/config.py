from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Role = Literal["router", "answer", "critic", "judge"]

DEFAULT_MODELS = {"anthropic": "claude-opus-5-5", "google": "gemini-3.5-flash"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # "offline" runs the whole graph with deterministic rule-based agents:
    # no API key, used by tests and CI.
    llm_provider: Literal["anthropic", "google", "offline"] = "offline"
    anthropic_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None
    # Client-side throttle, e.g. to stay inside a free-tier requests-per-minute quota.
    llm_requests_per_minute: float | None = None
    # Hard wall-clock cap per LLM step, enforced by the graph on top of the SDK.
    llm_timeout_s: float = 90.0
    environment: str = "development"

    langfuse_public_key: str | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"
    langfuse_prompt_label: str = "production"

    # Per-role model overrides; unset means the provider default (DEFAULT_MODELS).
    model_router: str | None = None
    model_answer: str | None = None
    model_critic: str | None = None
    model_judge: str | None = None
    # Anthropic only: effort per role.
    effort_router: Literal["low", "medium", "high"] = "low"
    effort_answer: Literal["low", "medium", "high"] = "medium"
    effort_critic: Literal["low", "medium", "high"] = "low"
    effort_judge: Literal["low", "medium", "high"] = "medium"

    # "local": the API process owns the vector store (single-process mode).
    # "grpc": the store lives in the retrieval service at retrieval_grpc_target.
    retrieval_backend: Literal["local", "grpc"] = "local"
    retrieval_grpc_target: str = "localhost:50051"
    grpc_port: int = 50051

    qdrant_url: str = ":memory:"
    collection: str = "documents"
    embedding_dim: int = 1024
    chunk_size: int = 700
    chunk_overlap: int = 100
    top_k: int = 4

    max_retrieval_rounds: int = 2
    retrieval_timeout_s: float = 10.0
    seed_corpus_dir: str | None = None

    def model_for(self, role: Role) -> str:
        return getattr(self, f"model_{role}") or DEFAULT_MODELS[self.llm_provider]


@lru_cache
def get_settings() -> Settings:
    return Settings()
