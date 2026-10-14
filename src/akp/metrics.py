"""Prometheus metrics. LLM-level detail (tokens, cost, prompts) lives in Langfuse;
these cover the infrastructure around it."""

import functools
import time

import grpc
from prometheus_client import Counter, Gauge, Histogram

ASK_REQUESTS = Counter("akp_ask_requests_total", "Answered /ask runs by final status", ["status"])
ASK_ERRORS = Counter("akp_ask_errors_total", "/ask runs that failed", ["kind"])
ASK_SECONDS = Histogram(
    "akp_ask_seconds",
    "End-to-end /ask latency",
    buckets=(0.5, 1, 2.5, 5, 10, 20, 40, 80, 160),
)

RETRIEVAL_REQUESTS = Counter(
    "akp_retrieval_requests_total", "gRPC calls to the retrieval service", ["method", "code"]
)
RETRIEVAL_SECONDS = Histogram(
    "akp_retrieval_seconds",
    "gRPC handler latency",
    ["method"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)

INGEST_MESSAGES = Counter(
    "akp_ingest_messages_total", "Ingestion messages handled, by outcome", ["outcome"]
)
INGEST_PENDING = Gauge("akp_ingest_pending", "Messages waiting for the ingest worker (lag)")


def observed_rpc(method):
    """Records latency and status code of a gRPC servicer method."""

    @functools.wraps(method)
    def wrapper(self, request, context):
        name = method.__name__
        started = time.perf_counter()
        code = "OK"
        try:
            return method(self, request, context)
        except Exception:
            status = context.code() if hasattr(context, "code") else None
            code = status.name if isinstance(status, grpc.StatusCode) else "UNKNOWN"
            raise
        finally:
            RETRIEVAL_SECONDS.labels(name).observe(time.perf_counter() - started)
            RETRIEVAL_REQUESTS.labels(name, code).inc()

    return wrapper
