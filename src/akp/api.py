import asyncio
import logging
import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager

import anthropic
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from google.genai.errors import APIError as GoogleAPIError
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from langgraph.errors import NodeTimeoutError
from nats.errors import Error as NatsError
from prometheus_client import make_asgi_app
from pydantic import BaseModel, Field

from akp.agents import Agents, make_agents
from akp.checkpoints import open_checkpointer
from akp.config import Settings, get_settings
from akp.graph import Status, build_graph, passages_of
from akp.ingestion import IngestAccepted, IngestPublisher, IngestRequest
from akp.metrics import ASK_ERRORS, ASK_REQUESTS, ASK_SECONDS
from akp.observability import Observability
from akp.retrieval import (
    DocumentInfo,
    DocumentStore,
    IngestResult,
    PassageRetriever,
    load_directory,
)
from akp.retrieval_service import RetrievalClient, RetrievalUnavailable

log = logging.getLogger(__name__)

ThreadId = Field(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = ThreadId


class Citation(BaseModel):
    title: str
    doc_id: str
    chunk_id: str
    text: str


class AskResponse(BaseModel):
    answer: str
    status: Status
    citations: list[Citation]
    search_queries: list[str]
    thread_id: str
    trace_id: str | None = None
    trace_url: str | None = None


class RunFailed(Exception):
    """A graph run stopped on an error; carries what the client needs to resume it."""

    def __init__(self, cause: Exception, thread_id: str, status_code: int, detail: str):
        super().__init__(str(cause))
        self.cause, self.thread_id = cause, thread_id
        self.status_code, self.detail = status_code, detail


_PROVIDER_ERRORS = (anthropic.APIError, GoogleAPIError, ChatGoogleGenerativeAIError)


def create_app(settings: Settings | None = None, agents: Agents | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with AsyncExitStack() as stack:
            if settings.retrieval_backend == "grpc":
                store = RetrievalClient(
                    settings.retrieval_grpc_target, settings.retrieval_timeout_s
                )
            else:
                store = DocumentStore.from_settings(settings)
                if settings.seed_corpus_dir:
                    load_directory(store, settings.seed_corpus_dir)
            retriever = PassageRetriever(backend=store, k=settings.top_k)
            obs = Observability.from_settings(settings)
            checkpointer = await open_checkpointer(settings, stack)
            publisher = None
            if settings.ingestion == "nats":
                publisher = await IngestPublisher.connect(settings)
                stack.push_async_callback(publisher.close)
            app.state.publisher = publisher
            app.state.store = store
            app.state.obs = obs
            app.state.resumable = checkpointer is not None
            app.state.graph = build_graph(
                agents or make_agents(settings, obs), retriever, settings, checkpointer
            )
            yield
            obs.flush()

    app = FastAPI(title="Agentic Knowledge Pipeline", lifespan=lifespan)
    app.mount("/metrics", make_asgi_app())

    @app.exception_handler(RunFailed)
    async def run_failed(request: Request, exc: RunFailed) -> JSONResponse:
        log.warning("run %s failed: %r", exc.thread_id, exc.cause)
        content = {"detail": exc.detail, "thread_id": exc.thread_id}
        if request.app.state.resumable:
            content["resume"] = f"/ask/{exc.thread_id}/resume"
        return JSONResponse(status_code=exc.status_code, content=content)

    @app.exception_handler(RetrievalUnavailable)
    async def retrieval_unavailable(request: Request, exc: RetrievalUnavailable) -> JSONResponse:
        log.warning("retrieval service unavailable: %s", exc)
        return JSONResponse(status_code=503, content={"detail": "Retrieval service is unavailable"})

    @app.exception_handler(NatsError)
    async def queue_unavailable(request: Request, exc: NatsError) -> JSONResponse:
        log.warning("ingestion queue unavailable: %r", exc)
        return JSONResponse(status_code=503, content={"detail": "Ingestion queue is unavailable"})

    async def run_graph(request: Request, graph_input, thread_id: str) -> AskResponse:
        obs: Observability = request.app.state.obs
        trace_id, callbacks = obs.new_trace()
        config = {
            "callbacks": callbacks,
            "run_name": "ask",
            "configurable": {"thread_id": thread_id},
        }
        started = time.perf_counter()
        try:
            state = await request.app.state.graph.ainvoke(graph_input, config)
        except _PROVIDER_ERRORS as exc:
            ASK_ERRORS.labels("llm_provider").inc()
            raise RunFailed(exc, thread_id, 502, "LLM provider error") from exc
        except NodeTimeoutError as exc:
            ASK_ERRORS.labels("llm_timeout").inc()
            raise RunFailed(exc, thread_id, 504, "LLM did not respond in time") from exc
        finally:
            ASK_SECONDS.observe(time.perf_counter() - started)
        ASK_REQUESTS.labels(state["status"]).inc()
        return AskResponse(
            answer=state["answer"],
            status=state["status"],
            citations=[Citation(**c.model_dump()) for c in passages_of(state, "citations")],
            search_queries=state.get("tried_queries", []),
            thread_id=thread_id,
            trace_id=trace_id,
            trace_url=obs.trace_url(trace_id),
        )

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "llm_provider": settings.llm_provider,
            "tracing": app.state.obs.enabled,
            "checkpointer": settings.checkpointer,
            "ingestion": settings.ingestion,
        }

    @app.post("/ask", response_model=AskResponse)
    async def ask(body: AskRequest, request: Request) -> AskResponse:
        thread_id = body.thread_id or uuid.uuid4().hex
        return await run_graph(request, {"question": body.question}, thread_id)

    @app.post("/ask/{thread_id}/resume", response_model=AskResponse)
    async def resume(thread_id: str, request: Request) -> AskResponse:
        """Continue a failed run from the step that failed; finished steps are not re-run."""
        if not request.app.state.resumable:
            raise HTTPException(status_code=404, detail="Checkpointing is disabled")
        snapshot = await request.app.state.graph.aget_state(
            {"configurable": {"thread_id": thread_id}}
        )
        if not snapshot.values:
            raise HTTPException(status_code=404, detail="Unknown thread")
        if not snapshot.next:
            raise HTTPException(status_code=409, detail="This run already finished")
        return await run_graph(request, None, thread_id)

    @app.post(
        "/documents",
        response_model=IngestResult | IngestAccepted,
        responses={202: {"model": IngestAccepted, "description": "Queued for ingestion"}},
    )
    async def add_document(
        body: IngestRequest, request: Request, response: Response
    ) -> IngestResult | IngestAccepted:
        if request.app.state.publisher:
            response.status_code = 202
            return await request.app.state.publisher.publish(body)
        result = await asyncio.to_thread(
            request.app.state.store.upsert_document, body.title, body.text
        )
        response.status_code = 201 if result.created else 200
        return result

    @app.get("/documents", response_model=list[DocumentInfo])
    def list_documents(request: Request) -> list[DocumentInfo]:
        return request.app.state.store.list_documents()

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str, request: Request) -> None:
        if not request.app.state.store.delete_document(doc_id):
            raise HTTPException(status_code=404, detail="Document not found")

    return app


app = create_app()
