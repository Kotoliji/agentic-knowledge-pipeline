import pytest
from fastapi.testclient import TestClient

from akp.api import create_app
from akp.config import Settings
from tests.conftest import CORPUS_DIR


@pytest.fixture
def client():
    settings = Settings(llm_provider="offline", seed_corpus_dir=str(CORPUS_DIR), _env_file=None)
    with TestClient(create_app(settings)) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "llm_provider": "offline"}


def test_ask_returns_answer_with_citations(client):
    body = client.post("/ask", json={"question": "How long does each on-call rotation last?"}).json()
    assert body["status"] == "answered"
    assert "one week" in body["answer"]
    assert body["citations"][0]["title"] == "On-Call Rotation"
    assert body["search_queries"] == ["How long does each on-call rotation last?"]


def test_ask_rejects_empty_question(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_document_lifecycle(client):
    doc = {"title": "Parking", "text": "Each employee gets one parking permit for level B2."}
    created = client.post("/documents", json=doc)
    assert created.status_code == 201
    doc_id = created.json()["doc_id"]

    duplicate = client.post("/documents", json=doc)
    assert duplicate.status_code == 200
    assert duplicate.json()["created"] is False

    titles = [d["title"] for d in client.get("/documents").json()]
    assert "Parking" in titles and len(titles) == 8

    answer = client.post("/ask", json={"question": "Which level is the parking permit for?"})
    assert answer.json()["citations"][0]["doc_id"] == doc_id

    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.delete(f"/documents/{doc_id}").status_code == 404
