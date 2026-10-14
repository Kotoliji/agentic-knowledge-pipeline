from pathlib import Path

import pytest

from akp.config import Settings
from akp.retrieval import DocumentStore, load_directory

CORPUS_DIR = Path(__file__).resolve().parent.parent / "corpus"


@pytest.fixture
def settings() -> Settings:
    return Settings(llm_provider="offline", qdrant_url=":memory:", _env_file=None)


@pytest.fixture
def store(settings: Settings) -> DocumentStore:
    return DocumentStore.from_settings(settings)


@pytest.fixture
def loaded_store(store: DocumentStore) -> DocumentStore:
    load_directory(store, CORPUS_DIR)
    return store
