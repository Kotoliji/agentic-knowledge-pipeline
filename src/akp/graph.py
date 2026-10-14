from typing import Literal, TypedDict

from langchain_core.exceptions import OutputParserException
from langgraph.errors import NodeTimeoutError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy
from pydantic import ValidationError

from akp.agents import Agents, Critique, Draft
from akp.config import Settings
from akp.retrieval import Passage, PassageRetriever, passage_from_document

Status = Literal["answered", "insufficient_evidence", "direct", "retrieval_failed"]

NOT_FOUND = "I couldn't find an answer to that in the documents."
RETRIEVAL_DOWN = "Document search is temporarily unavailable, so I can't answer right now."


class AgentState(TypedDict, total=False):
    question: str
    route: str
    query: str
    tried_queries: list[str]
    # Models are stored as plain dicts (model_dump) and rebuilt on read: the Redis
    # checkpointer deliberately refuses to revive arbitrary classes from storage, so
    # state must be JSON-native to survive a resume.
    passages: list[dict]
    draft: dict
    critique: dict
    status: Status | None
    answer: str
    citations: list[dict]


def passages_of(state: AgentState, key: str = "passages") -> list[Passage]:
    return [Passage.model_validate(p) for p in state.get(key) or []]


def draft_of(state: AgentState) -> Draft:
    return Draft.model_validate(state["draft"])


def critique_of(state: AgentState) -> Critique:
    return Critique.model_validate(state["critique"])


def _valid_citations(draft: Draft, n_passages: int) -> Draft:
    """Never trust model-produced indices: drop out-of-range ones, and an answer
    left with no valid citation is treated as unanswerable."""
    citations = sorted({i for i in draft.citations if 1 <= i <= n_passages})
    return draft.model_copy(
        update={"citations": citations, "answerable": draft.answerable and bool(citations)}
    )


def build_graph(agents: Agents, retriever: PassageRetriever, settings: Settings, checkpointer=None):
    async def route(state: AgentState) -> dict:
        decision = await agents.route(state["question"])
        # A checkpointed thread keeps its state between questions: reset everything a
        # previous run may have left behind (a stale status would short-circuit finalize).
        return {
            "route": decision.route,
            "query": decision.search_query or state["question"],
            "tried_queries": [],
            "passages": [],
            "status": None,
            "answer": "",
            "citations": [],
        }

    async def retrieve(state: AgentState) -> dict:
        docs = await retriever.ainvoke(state["query"])
        return {
            "passages": [passage_from_document(d).model_dump() for d in docs],
            "tried_queries": [*state.get("tried_queries", []), state["query"]],
        }

    def retrieval_failed(state: AgentState) -> Command:
        # Called by LangGraph once the retry policy is exhausted: degrade to an
        # explicit "unavailable" answer instead of failing the whole request.
        return Command(
            update={"status": "retrieval_failed", "answer": RETRIEVAL_DOWN, "citations": []},
            goto="finalize",
        )

    async def draft(state: AgentState) -> dict:
        passages = passages_of(state)
        result = await agents.draft(state["question"], passages)
        return {"draft": _valid_citations(result, len(passages)).model_dump()}

    async def critic(state: AgentState) -> dict:
        critique = await agents.critique(state["question"], draft_of(state), passages_of(state))
        return {"critique": critique.model_dump()}

    def rewrite_query(state: AgentState) -> dict:
        return {"query": critique_of(state).better_query}

    async def direct(state: AgentState) -> dict:
        return {"status": "direct", "answer": await agents.chat(state["question"]), "citations": []}

    def finalize(state: AgentState) -> dict:
        if state.get("status") == "retrieval_failed":
            return {}
        draft, critique = draft_of(state), critique_of(state)
        if draft.answerable and critique.supported:
            passages = passages_of(state)
            cited = [passages[i - 1].model_dump() for i in draft.citations]
            return {"status": "answered", "answer": draft.answer, "citations": cited}
        return {"status": "insufficient_evidence", "answer": NOT_FOUND, "citations": []}

    def after_route(state: AgentState) -> str:
        return "direct" if state["route"] == "direct" else "retrieve"

    def after_critic(state: AgentState) -> str:
        draft, critique = draft_of(state), critique_of(state)
        if draft.answerable and critique.supported:
            return "finalize"
        retry = critique.better_query.strip()
        tried = state["tried_queries"]
        if retry and retry not in tried and len(tried) < settings.max_retrieval_rounds:
            return "rewrite_query"
        return "finalize"

    # The SDKs already retry transport errors (429/5xx). The graph adds what they
    # can't: a hard per-step timeout, and one retry when the model returns output
    # that doesn't parse into the schema. Retrying both layers on the same errors
    # would multiply into very long stalls.
    llm_step = {
        "retry_policy": RetryPolicy(
            max_attempts=2, retry_on=(NodeTimeoutError, OutputParserException, ValidationError)
        ),
        "timeout": settings.llm_timeout_s,
    }
    graph = StateGraph(AgentState)
    graph.add_node("route", route, **llm_step)
    graph.add_node(
        "retrieve",
        retrieve,
        retry_policy=RetryPolicy(
            max_attempts=3,
            initial_interval=0.2,
            retry_on=(ConnectionError, TimeoutError, NodeTimeoutError),
        ),
        # Longer than the gRPC deadline, so the client's own error surfaces first.
        timeout=settings.retrieval_timeout_s + 2,
        error_handler=retrieval_failed,
        destinations=("finalize",),
    )
    graph.add_node("draft", draft, **llm_step)
    graph.add_node("critic", critic, **llm_step)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("direct", direct, **llm_step)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", after_route, ["retrieve", "direct"])
    graph.add_edge("retrieve", "draft")
    graph.add_edge("draft", "critic")
    graph.add_conditional_edges("critic", after_critic, ["rewrite_query", "finalize"])
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("direct", END)
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)
