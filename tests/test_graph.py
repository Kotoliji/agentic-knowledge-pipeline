import pytest

from akp.agents import Critique, Draft, OfflineAgents, RouteDecision
from akp.graph import NOT_FOUND, RETRIEVAL_DOWN, build_graph
from akp.retrieval import PassageRetriever


def make_graph(agents, backend, settings):
    return build_graph(agents, PassageRetriever(backend=backend, k=settings.top_k), settings)


async def ask(graph, question):
    return await graph.ainvoke({"question": question})


async def test_answers_with_citation_from_the_right_document(loaded_store, settings):
    graph = make_graph(OfflineAgents(), loaded_store, settings)
    state = await ask(graph, "What is the hotel limit per night in London?")
    assert state["status"] == "answered"
    assert "220 EUR" in state["answer"]
    assert [c.title for c in state["citations"]] == ["Expense Policy"]


async def test_question_outside_the_documents_is_not_answered(loaded_store, settings):
    graph = make_graph(OfflineAgents(), loaded_store, settings)
    state = await ask(graph, "What was the company's revenue in 2024?")
    assert state["status"] == "insufficient_evidence"
    assert state["answer"] == NOT_FOUND
    assert state["citations"] == []


async def test_small_talk_skips_retrieval(loaded_store, settings):
    graph = make_graph(OfflineAgents(), loaded_store, settings)
    state = await ask(graph, "hi")
    assert state["status"] == "direct"
    assert "passages" not in state


class ScriptedAgents(OfflineAgents):
    """Offline agents whose critic verdicts are scripted per round."""

    def __init__(self, critiques):
        self.critiques = list(critiques)

    async def critique(self, question, draft, passages):
        return self.critiques.pop(0)


async def test_critic_rejection_triggers_a_second_retrieval_with_a_new_query(
    loaded_store, settings
):
    agents = ScriptedAgents(
        [
            Critique(supported=False, issues="wrong policy", better_query="parental leave weeks"),
            Critique(supported=True, issues="", better_query=""),
        ]
    )
    graph = make_graph(agents, loaded_store, settings)
    state = await ask(graph, "How many weeks of parental leave do birth parents get?")
    assert state["tried_queries"] == [
        "How many weeks of parental leave do birth parents get?",
        "parental leave weeks",
    ]
    assert state["status"] == "answered"


async def test_retrieval_rounds_are_bounded(loaded_store, settings):
    agents = ScriptedAgents(
        [Critique(supported=False, issues="", better_query=f"query {i}") for i in range(10)]
    )
    graph = make_graph(agents, loaded_store, settings)
    state = await ask(graph, "What is the hotel limit per night in London?")
    assert len(state["tried_queries"]) == settings.max_retrieval_rounds
    assert state["status"] == "insufficient_evidence"


async def test_out_of_range_citations_are_not_trusted(loaded_store, settings):
    class HallucinatedCitations(OfflineAgents):
        async def draft(self, question, passages):
            return Draft(answer="Made up.", citations=[99], answerable=True)

    graph = make_graph(HallucinatedCitations(), loaded_store, settings)
    state = await ask(graph, "What is the hotel limit per night in London?")
    assert state["draft"].citations == []
    assert state["draft"].answerable is False
    assert state["status"] == "insufficient_evidence"


class FlakyBackend:
    def __init__(self, inner, failures):
        self.inner, self.failures, self.calls = inner, failures, 0

    def search(self, query, k):
        self.calls += 1
        if self.calls <= self.failures:
            raise ConnectionError("retrieval backend unreachable")
        return self.inner.search(query, k)


async def test_transient_retrieval_failure_is_retried(loaded_store, settings):
    backend = FlakyBackend(loaded_store, failures=1)
    graph = make_graph(OfflineAgents(), backend, settings)
    state = await ask(graph, "What is the hotel limit per night in London?")
    assert backend.calls == 2
    assert state["status"] == "answered"


async def test_persistent_retrieval_failure_degrades_gracefully(loaded_store, settings):
    backend = FlakyBackend(loaded_store, failures=100)
    graph = make_graph(OfflineAgents(), backend, settings)
    state = await ask(graph, "What is the hotel limit per night in London?")
    assert backend.calls == 3
    assert state["status"] == "retrieval_failed"
    assert state["answer"] == RETRIEVAL_DOWN


async def test_router_decision_drives_the_search_query(loaded_store, settings):
    class RewritingRouter(OfflineAgents):
        async def route(self, question):
            return RouteDecision(route="retrieve", search_query="on-call stipend primary week")

    graph = make_graph(RewritingRouter(), loaded_store, settings)
    state = await ask(graph, "how much do I get paid extra for being on call?")
    assert state["tried_queries"] == ["on-call stipend primary week"]


@pytest.mark.parametrize("question", ["hello", "thanks"])
async def test_offline_router_small_talk(question):
    assert (await OfflineAgents().route(question)).route == "direct"


@pytest.mark.parametrize(
    ("provider", "expected_model"),
    [("anthropic", "claude-opus-5-5"), ("google", "gemini-2.5-flash")],
)
def test_llm_agents_wire_up_for_each_provider(settings, store, provider, expected_model):
    from akp.agents import LLMAgents, chat_model

    s = settings.model_copy(
        update={
            "llm_provider": provider,
            "anthropic_api_key": "test-key-not-used",
            "google_api_key": "test-key-not-used",
            "llm_requests_per_minute": 10,
        }
    )
    graph = make_graph(LLMAgents(s), store, s)
    assert set(graph.get_graph().nodes) >= {"route", "retrieve", "draft", "critic", "finalize"}
    assert s.model_for("router") == expected_model
    assert chat_model(s, "judge", 100).rate_limiter is not None


def test_per_role_model_override(settings):
    s = settings.model_copy(update={"llm_provider": "google", "model_judge": "gemini-x"})
    assert s.model_for("judge") == "gemini-x"
    assert s.model_for("answer") == "gemini-2.5-flash"
