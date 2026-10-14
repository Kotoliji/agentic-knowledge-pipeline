import logging
from contextlib import asynccontextmanager

import anthropic
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from google.genai.errors import APIError as GoogleAPIError
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from pydantic import BaseModel, Field

from akp.agents import Agents, make_agents
from akp.config import Settings, get_settings
from akp.graph import Status, build_graph
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


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


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
    trace_id: str | None = None
    trace_url: str | None = None


class DocumentIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=200_000)


def create_app(settings: Settings | None = None, agents: Agents | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.retrieval_backend == "grpc":
            store = RetrievalClient(settings.retrieval_grpc_target, settings.retrieval_timeout_s)
        else:
            store = DocumentStore.from_settings(settings)
            if settings.seed_corpus_dir:
                load_directory(store, settings.seed_corpus_dir)
        retriever = PassageRetriever(backend=store, k=settings.top_k)
        obs = Observability.from_settings(settings)
        app.state.store = store
        app.state.obs = obs
        app.state.graph = build_graph(agents or make_agents(settings, obs), retriever, settings)
        yield
        obs.flush()

    app = FastAPI(title="Agentic Knowledge Pipeline", lifespan=lifespan)

    async def llm_unavailable(request: Request, exc: Exception) -> JSONResponse:
        log.warning("LLM provider error: %s", exc)
        return JSONResponse(status_code=502, content={"detail": "LLM provider error"})

    for provider_error in (anthropic.APIError, GoogleAPIError, ChatGoogleGenerativeAIError):
        app.add_exception_handler(provider_error, llm_unavailable)

    @app.exception_handler(RetrievalUnavailable)
    async def retrieval_unavailable(request: Request, exc: RetrievalUnavailable) -> JSONResponse:
        log.warning("retrieval service unavailable: %s", exc)
        return JSONResponse(status_code=503, content={"detail": "Retrieval service is unavailable"})

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "llm_provider": settings.llm_provider,
            "tracing": app.state.obs.enabled,
        }

    @app.post("/ask", response_model=AskResponse)
    async def ask(body: AskRequest, request: Request) -> AskResponse:
        obs: Observability = request.app.state.obs
        trace_id, callbacks = obs.new_trace()
        state = await request.app.state.graph.ainvoke(
            {"question": body.question},
            {"callbacks": callbacks, "run_name": "ask"},
        )
        return AskResponse(
            answer=state["answer"],
            status=state["status"],
            citations=[Citation(**c.model_dump()) for c in state.get("citations", [])],
            search_queries=state.get("tried_queries", []),
            trace_id=trace_id,
            trace_url=obs.trace_url(trace_id),
        )

    @app.post("/documents", response_model=IngestResult)
    def add_document(body: DocumentIn, request: Request, response: Response) -> IngestResult:
        result = request.app.state.store.upsert_document(body.title, body.text)
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
