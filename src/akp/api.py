from contextlib import asynccontextmanager

import anthropic
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from akp.agents import Agents, make_agents
from akp.config import Settings, get_settings
from akp.graph import Status, build_graph
from akp.retrieval import (
    DocumentInfo,
    DocumentStore,
    IngestResult,
    PassageRetriever,
    load_directory,
)


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


class DocumentIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=200_000)


def create_app(settings: Settings | None = None, agents: Agents | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = DocumentStore.from_settings(settings)
        if settings.seed_corpus_dir:
            load_directory(store, settings.seed_corpus_dir)
        retriever = PassageRetriever(backend=store, k=settings.top_k)
        app.state.store = store
        app.state.graph = build_graph(agents or make_agents(settings), retriever, settings)
        yield

    app = FastAPI(title="Agentic Knowledge Pipeline", lifespan=lifespan)

    @app.exception_handler(anthropic.APIError)
    async def llm_unavailable(request: Request, exc: anthropic.APIError) -> JSONResponse:
        return JSONResponse(status_code=502, content={"detail": f"LLM provider error: {exc}"})

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "llm_provider": settings.llm_provider}

    @app.post("/ask", response_model=AskResponse)
    async def ask(body: AskRequest, request: Request) -> AskResponse:
        state = await request.app.state.graph.ainvoke({"question": body.question})
        return AskResponse(
            answer=state["answer"],
            status=state["status"],
            citations=[Citation(**c.model_dump()) for c in state.get("citations", [])],
            search_queries=state.get("tried_queries", []),
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
