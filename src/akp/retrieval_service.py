"""gRPC retrieval service: the single owner of the vector store.

Run the server with `python -m akp.retrieval_service`.
"""

import json
import logging
from concurrent import futures

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from akp.config import Settings, get_settings
from akp.metrics import observed_rpc
from akp.retrieval import DocumentInfo, DocumentStore, IngestResult, Passage, load_directory
from akp.rpc import retrieval_pb2 as pb
from akp.rpc import retrieval_pb2_grpc as pb_grpc

log = logging.getLogger(__name__)

SERVICE_NAME = "akp.retrieval.v1.RetrievalService"
MAX_TOP_K = 50


class RetrievalServicer(pb_grpc.RetrievalServiceServicer):
    def __init__(self, store: DocumentStore, default_top_k: int):
        self.store = store
        self.default_top_k = default_top_k

    @observed_rpc
    def Search(self, request, context):
        query = request.query.strip()
        if not query:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "query must not be empty")
        if request.top_k > MAX_TOP_K:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"top_k must be <= {MAX_TOP_K}")
        passages = self.store.search(query, request.top_k or self.default_top_k)
        return pb.SearchResponse(passages=[pb.Passage(**p.model_dump()) for p in passages])

    @observed_rpc
    def UpsertDocument(self, request, context):
        if not request.title.strip() or not request.text.strip():
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "title and text are required")
        result = self.store.upsert_document(request.title, request.text)
        return pb.UpsertDocumentResponse(**result.model_dump())

    @observed_rpc
    def ListDocuments(self, request, context):
        docs = self.store.list_documents()
        return pb.ListDocumentsResponse(documents=[pb.DocumentInfo(**d.model_dump()) for d in docs])

    @observed_rpc
    def DeleteDocument(self, request, context):
        if not self.store.delete_document(request.doc_id):
            context.abort(grpc.StatusCode.NOT_FOUND, f"document {request.doc_id} not found")
        return pb.DeleteDocumentResponse()


def create_server(
    store: DocumentStore, settings: Settings, address: str
) -> tuple[grpc.Server, int]:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    pb_grpc.add_RetrievalServiceServicer_to_server(RetrievalServicer(store, settings.top_k), server)
    health_servicer = health.HealthServicer()
    health_servicer.set(SERVICE_NAME, health_pb2.HealthCheckResponse.SERVING)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    port = server.add_insecure_port(address)
    return server, port


class RetrievalUnavailable(ConnectionError):
    """The service could not be reached or did not answer in time.

    Subclasses ConnectionError so the agent graph's retry policy treats it as
    transient and, once retries are exhausted, degrades instead of crashing.
    """


# gRPC-level retries: only UNAVAILABLE (connection refused, server restarting)
# is retried, never DEADLINE_EXCEEDED — the deadline is the caller's total budget.
_SERVICE_CONFIG = json.dumps(
    {
        "methodConfig": [
            {
                "name": [{"service": SERVICE_NAME}],
                "retryPolicy": {
                    "maxAttempts": 3,
                    "initialBackoff": "0.1s",
                    "maxBackoff": "1s",
                    "backoffMultiplier": 2,
                    "retryableStatusCodes": ["UNAVAILABLE"],
                },
            }
        ]
    }
)


class RetrievalClient:
    """Implements the same interface as DocumentStore, over gRPC."""

    def __init__(self, target: str, timeout_s: float):
        self.timeout_s = timeout_s
        self._channel = grpc.insecure_channel(
            target,
            options=[("grpc.enable_retries", 1), ("grpc.service_config", _SERVICE_CONFIG)],
        )
        self._stub = pb_grpc.RetrievalServiceStub(self._channel)

    def _call(self, method, request):
        try:
            return method(request, timeout=self.timeout_s)
        except grpc.RpcError as err:
            code = err.code()
            if code in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED):
                raise RetrievalUnavailable(
                    f"retrieval service {code.name}: {err.details()}"
                ) from err
            if code == grpc.StatusCode.INVALID_ARGUMENT:
                raise ValueError(err.details()) from err
            raise

    def search(self, query: str, k: int) -> list[Passage]:
        response = self._call(self._stub.Search, pb.SearchRequest(query=query, top_k=k))
        return [
            Passage(chunk_id=p.chunk_id, doc_id=p.doc_id, title=p.title, text=p.text, score=p.score)
            for p in response.passages
        ]

    def upsert_document(self, title: str, text: str) -> IngestResult:
        r = self._call(self._stub.UpsertDocument, pb.UpsertDocumentRequest(title=title, text=text))
        return IngestResult(doc_id=r.doc_id, title=r.title, chunks=r.chunks, created=r.created)

    def list_documents(self) -> list[DocumentInfo]:
        r = self._call(self._stub.ListDocuments, pb.ListDocumentsRequest())
        return [DocumentInfo(doc_id=d.doc_id, title=d.title, chunks=d.chunks) for d in r.documents]

    def delete_document(self, doc_id: str) -> bool:
        try:
            self._call(self._stub.DeleteDocument, pb.DeleteDocumentRequest(doc_id=doc_id))
            return True
        except grpc.RpcError as err:
            if err.code() == grpc.StatusCode.NOT_FOUND:
                return False
            raise

    def close(self) -> None:
        self._channel.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    store = DocumentStore.from_settings(settings)
    if settings.seed_corpus_dir:
        load_directory(store, settings.seed_corpus_dir)
    server, port = create_server(store, settings, f"0.0.0.0:{settings.grpc_port}")
    server.start()
    if settings.metrics_port:
        from prometheus_client import start_http_server

        start_http_server(settings.metrics_port)
    log.info("retrieval service listening on :%d", port)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
