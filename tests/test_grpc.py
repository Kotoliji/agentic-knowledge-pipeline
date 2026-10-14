import socket
import time

import grpc
import pytest
from fastapi.testclient import TestClient
from grpc_health.v1 import health_pb2, health_pb2_grpc

from akp.agents import OfflineAgents
from akp.api import create_app
from akp.config import Settings
from akp.graph import RETRIEVAL_DOWN, build_graph
from akp.retrieval import PassageRetriever
from akp.retrieval_service import (
    SERVICE_NAME,
    RetrievalClient,
    RetrievalServicer,
    RetrievalUnavailable,
    create_server,
)


@pytest.fixture
def grpc_target(loaded_store, settings):
    server, port = create_server(loaded_store, settings, "127.0.0.1:0")
    server.start()
    yield f"127.0.0.1:{port}"
    server.stop(grace=0)


@pytest.fixture
def client(grpc_target):
    c = RetrievalClient(grpc_target, timeout_s=5)
    yield c
    c.close()


def closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_health_check_reports_serving(grpc_target):
    with grpc.insecure_channel(grpc_target) as channel:
        stub = health_pb2_grpc.HealthStub(channel)
        response = stub.Check(health_pb2.HealthCheckRequest(service=SERVICE_NAME), timeout=5)
    assert response.status == health_pb2.HealthCheckResponse.SERVING


def test_search(client):
    passages = client.search("hotel limit per night in London", k=2)
    assert len(passages) == 2
    assert passages[0].title == "Expense Policy"
    assert passages[0].score > 0


def test_upsert_is_idempotent_and_documents_can_be_deleted(client):
    first = client.upsert_document("Parking", "Permits are for level B2.")
    again = client.upsert_document("Parking", "Permits are for level B2.")
    assert first.created and not again.created
    assert "Parking" in {d.title for d in client.list_documents()}
    assert client.delete_document(first.doc_id) is True
    assert client.delete_document(first.doc_id) is False


@pytest.mark.parametrize(("query", "k"), [("   ", 3), ("hotel", 51)])
def test_invalid_arguments_are_rejected(client, query, k):
    with pytest.raises(ValueError):
        client.search(query, k)


def test_unreachable_service_raises_retrieval_unavailable():
    client = RetrievalClient(f"127.0.0.1:{closed_port()}", timeout_s=1)
    with pytest.raises(RetrievalUnavailable, match="UNAVAILABLE|DEADLINE_EXCEEDED"):
        client.search("hotel", 3)


def test_slow_server_hits_the_deadline(loaded_store, settings):
    class SlowServicer(RetrievalServicer):
        def Search(self, request, context):
            time.sleep(1)
            return super().Search(request, context)

    server = grpc.server(__import__("concurrent.futures").futures.ThreadPoolExecutor(2))
    from akp.rpc import retrieval_pb2_grpc

    retrieval_pb2_grpc.add_RetrievalServiceServicer_to_server(SlowServicer(loaded_store, 4), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        client = RetrievalClient(f"127.0.0.1:{port}", timeout_s=0.2)
        with pytest.raises(RetrievalUnavailable, match="DEADLINE_EXCEEDED"):
            client.search("hotel", 3)
    finally:
        server.stop(grace=0)


async def test_agent_graph_answers_through_the_grpc_service(client, settings):
    graph = build_graph(OfflineAgents(), PassageRetriever(backend=client), settings)
    state = await graph.ainvoke({"question": "What is the hotel limit per night in London?"})
    assert state["status"] == "answered"
    assert "220 EUR" in state["answer"]


async def test_agent_graph_degrades_when_the_service_is_down(settings):
    client = RetrievalClient(f"127.0.0.1:{closed_port()}", timeout_s=0.5)
    graph = build_graph(OfflineAgents(), PassageRetriever(backend=client), settings)
    state = await graph.ainvoke({"question": "What is the hotel limit per night in London?"})
    assert state["status"] == "retrieval_failed"
    assert state["answer"] == RETRIEVAL_DOWN


def test_api_in_grpc_mode(grpc_target):
    settings = Settings(retrieval_backend="grpc", retrieval_grpc_target=grpc_target, _env_file=None)
    with TestClient(create_app(settings)) as api:
        assert len(api.get("/documents").json()) == 7
        body = api.post("/ask", json={"question": "How long does each on-call rotation last?"})
        assert body.json()["status"] == "answered"


def test_api_returns_503_when_retrieval_service_is_down():
    settings = Settings(
        retrieval_backend="grpc",
        retrieval_grpc_target=f"127.0.0.1:{closed_port()}",
        retrieval_timeout_s=0.5,
        _env_file=None,
    )
    with TestClient(create_app(settings)) as api:
        assert api.get("/documents").status_code == 503


def test_grpc_metrics_record_codes(client):
    from prometheus_client import REGISTRY

    def count(code):
        value = REGISTRY.get_sample_value(
            "akp_retrieval_requests_total", {"method": "Search", "code": code}
        )
        return value or 0

    ok, bad = count("OK"), count("INVALID_ARGUMENT")
    client.search("hotel", 2)
    with pytest.raises(ValueError):
        client.search("   ", 2)
    assert count("OK") == ok + 1
    assert count("INVALID_ARGUMENT") == bad + 1
