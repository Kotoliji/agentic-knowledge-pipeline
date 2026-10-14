import asyncio
import os
from contextlib import AsyncExitStack

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from akp.agents import OfflineAgents
from akp.api import create_app
from akp.checkpoints import open_checkpointer
from akp.config import Settings
from akp.graph import build_graph
from akp.retrieval import PassageRetriever
from tests.conftest import CORPUS_DIR

QUESTION = "What is the hotel limit per night in London?"


class CriticFailsOnce(OfflineAgents):
    """Counts calls per step; the critic raises on its first call (a provider outage)."""

    def __init__(self):
        self.calls = {"route": 0, "draft": 0, "critique": 0}

    async def route(self, question):
        self.calls["route"] += 1
        return await super().route(question)

    async def draft(self, question, passages):
        self.calls["draft"] += 1
        return await super().draft(question, passages)

    async def critique(self, question, draft, passages):
        self.calls["critique"] += 1
        if self.calls["critique"] == 1:
            raise RuntimeError("provider down")
        return await super().critique(question, draft, passages)


async def _resume_after_failure(checkpointer, loaded_store, settings):
    agents = CriticFailsOnce()
    graph = build_graph(agents, PassageRetriever(backend=loaded_store), settings, checkpointer)
    config = {"configurable": {"thread_id": f"t-{id(agents)}"}}

    with pytest.raises(RuntimeError, match="provider down"):
        await graph.ainvoke({"question": QUESTION}, config)
    assert (await graph.aget_state(config)).next == ("critic",)

    state = await graph.ainvoke(None, config)
    assert state["status"] == "answered"
    # Steps that had finished before the failure were not run (or paid for) again.
    assert agents.calls == {"route": 1, "draft": 1, "critique": 2}


async def test_failed_run_resumes_from_the_failed_step(loaded_store, settings):
    await _resume_after_failure(InMemorySaver(), loaded_store, settings)


async def test_a_new_question_on_the_same_thread_starts_clean(loaded_store, settings):
    class Toggle:
        down = True

        def search(self, query, k):
            if self.down:
                raise ConnectionError("down")
            return loaded_store.search(query, k)

    backend = Toggle()
    s = settings.model_copy(update={"retrieval_timeout_s": 1})
    graph = build_graph(OfflineAgents(), PassageRetriever(backend=backend), s, InMemorySaver())
    config = {"configurable": {"thread_id": "same"}}
    assert (await graph.ainvoke({"question": QUESTION}, config))["status"] == "retrieval_failed"
    backend.down = False
    assert (await graph.ainvoke({"question": QUESTION}, config))["status"] == "answered"


def test_api_reports_a_resumable_failure_and_resumes_it():
    agents = CriticFailsOnce()
    settings = Settings(seed_corpus_dir=str(CORPUS_DIR), llm_timeout_s=0.3, _env_file=None)

    async def hangs_once(question, draft, passages):
        agents.calls["critique"] += 1
        if agents.calls["critique"] <= 2:  # both graph attempts time out
            await asyncio.sleep(5)
        return await OfflineAgents.critique(agents, question, draft, passages)

    agents.critique = hangs_once
    with TestClient(create_app(settings, agents)) as client:
        failed = client.post("/ask", json={"question": QUESTION})
        assert failed.status_code == 504
        thread_id = failed.json()["thread_id"]
        assert failed.json()["resume"] == f"/ask/{thread_id}/resume"

        resumed = client.post(f"/ask/{thread_id}/resume")
        assert resumed.status_code == 200
        assert resumed.json()["status"] == "answered"
        assert agents.calls["route"] == 1 and agents.calls["draft"] == 1

        assert client.post(f"/ask/{thread_id}/resume").status_code == 409
        assert client.post("/ask/no-such-thread/resume").status_code == 404


@pytest.mark.skipif(not os.environ.get("AKP_TEST_REDIS_URL"), reason="set AKP_TEST_REDIS_URL")
async def test_resume_with_the_redis_checkpointer(loaded_store, settings):
    s = settings.model_copy(
        update={"checkpointer": "redis", "redis_url": os.environ["AKP_TEST_REDIS_URL"]}
    )
    async with AsyncExitStack() as stack:
        saver = await open_checkpointer(s, stack)
        await _resume_after_failure(saver, loaded_store, s)
