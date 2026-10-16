import os

import pytest

from akp.agents import OfflineAgents
from akp.evaluation import (
    EvalItem,
    EvalRecord,
    HeuristicJudge,
    HumanLabel,
    LLMJudge,
    RetrievalItem,
    Verdict,
    agreement,
    cohen_kappa,
    load_items,
    load_retrieval_sets,
    read_records,
    retrieval_metrics,
    run_eval,
    summarize,
)
from akp.graph import build_graph
from akp.observability import Observability
from akp.retrieval import DocumentStore, PassageRetriever, load_directory
from tests.conftest import CORPUS_DIR

DATASET = CORPUS_DIR.parent / "evals" / "dataset.jsonl"


def test_dataset_is_well_formed():
    items = load_items(DATASET)
    assert len(items) == 30
    assert len({i.id for i in items}) == 30
    assert sum(not i.answerable for i in items) == 6
    titles = {p.read_text().splitlines()[0][2:] for p in CORPUS_DIR.glob("*.md")}
    assert {i.source for i in items if i.answerable} <= titles


async def test_offline_run_over_the_whole_dataset(loaded_store, settings):
    graph = build_graph(OfflineAgents(), PassageRetriever(backend=loaded_store), settings)
    records = await run_eval(load_items(DATASET), graph, HeuristicJudge(), Observability(), "t")
    summary = summarize(records)

    assert summary["graded_items"] == 30
    assert summary["errored_items"] == 0
    assert sum(summary["statuses"].values()) == 30
    # The keyword-matching offline agents do "answer" some unanswerable questions
    # with an unrelated sentence; the eval must catch exactly those.
    for r in records:
        if not r.item.answerable:
            assert r.verdict.correct == (r.status != "answered")
    assert 0.2 < summary["correct"] < 1.0


@pytest.mark.parametrize(
    ("pairs", "expected"),
    [
        ([(True, True), (False, False)] * 5, 1.0),
        ([(True, False), (False, True)] * 5, -1.0),
        ([(True, True)] * 10, None),  # no negatives on either side: undefined
        # 20 items: both say yes 14 times, both no 2, disagree 4 -> po=0.8, pe=0.68
        (
            [(True, True)] * 14 + [(False, False)] * 2 + [(True, False)] * 2 + [(False, True)] * 2,
            (0.8 - 0.68) / (1 - 0.68),
        ),
    ],
)
def test_cohen_kappa(pairs, expected):
    kappa = cohen_kappa(pairs)
    assert kappa is None if expected is None else kappa == pytest.approx(expected)


def _record(item_id: str, correct: bool, faithful: bool) -> EvalRecord:
    return EvalRecord(
        item=EvalItem(id=item_id, question="q", reference="r", answerable=True),
        answer="a",
        status="answered",
        citations=[],
        search_queries=["q"],
        trace_id=None,
        verdict=Verdict(reasoning="", correct=correct, faithful=faithful),
    )


def test_agreement_reports_disagreements_and_ignores_unlabelled():
    records = [_record("a", True, True), _record("b", True, False), _record("c", False, True)]
    labels = [
        HumanLabel(id="a", correct=True, faithful=True),
        HumanLabel(id="b", correct=False, faithful=False),
    ]
    report = agreement(records, labels)
    assert report["labelled_items"] == 2
    assert report["correct"]["agreement"] == 0.5
    assert report["correct"]["judge_yes_human_no"] == 1
    assert report["correct"]["disagreements"] == ["b"]
    assert report["faithful"]["agreement"] == 1.0


def test_llm_judge_wires_up_without_calling_the_api(monkeypatch, settings):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    LLMJudge(settings.model_copy(update={"llm_provider": "anthropic"}), Observability())


class FailingJudge(HeuristicJudge):
    def __init__(self, fail_ids):
        self.fail_ids = set(fail_ids)

    async def grade(self, item, answer, citations, config):
        if item.id in self.fail_ids:
            raise RuntimeError("quota exhausted")
        return await super().grade(item, answer, citations, config)


async def test_a_failing_item_is_recorded_and_the_run_continues(loaded_store, settings):
    graph = build_graph(OfflineAgents(), PassageRetriever(backend=loaded_store), settings)
    items = load_items(DATASET)[:5]
    persisted = []
    records = await run_eval(
        items, graph, FailingJudge({items[1].id}), Observability(), "t", 1, persisted.append
    )
    assert len(records) == len(persisted) == 5
    failed = next(r for r in records if r.item.id == items[1].id)
    assert failed.verdict is None and "quota exhausted" in failed.error
    summary = summarize(records)
    assert summary["graded_items"] == 4 and summary["errored_items"] == 1


async def test_consecutive_failures_stop_the_run_for_a_later_resume(loaded_store, settings):
    graph = build_graph(OfflineAgents(), PassageRetriever(backend=loaded_store), settings)
    items = load_items(DATASET)[:10]
    judge = FailingJudge({i.id for i in items})
    records = await run_eval(items, graph, judge, Observability(), "t", 1)
    assert len(records) == 3  # the other 7 are left for --resume


def test_read_records_keeps_the_latest_record_per_item(tmp_path):
    failed = _record("a", True, True).model_copy(update={"verdict": None, "error": "429"})
    path = tmp_path / "run.jsonl"
    path.write_text(failed.model_dump_json() + "\n" + _record("a", True, False).model_dump_json())
    [record] = read_records(path)
    assert record.verdict is not None and record.verdict.faithful is False


def test_paraphrase_set_is_well_formed():
    sets = load_retrieval_sets()
    titles = {p.read_text().splitlines()[0][2:] for p in CORPUS_DIR.glob("*.md")}
    paraphrases = sets["paraphrases"]
    assert len({i.id for i in paraphrases}) == len(paraphrases) >= 10
    assert {i.source for i in paraphrases} <= titles


def test_retrieval_metrics(loaded_store):
    items = [
        RetrievalItem(
            id="hit", question="hotel limit per night in London", source="Expense Policy"
        ),
        RetrievalItem(id="miss", question="hotel limit per night", source="No Such Document"),
    ]
    metrics = retrieval_metrics(loaded_store, items, k=4)
    assert metrics["questions"] == 2
    assert metrics["recall@1"] == 0.5
    assert metrics["mrr"] == 0.5
    assert metrics["not_first"] == ["miss"]


@pytest.mark.skipif(
    not os.environ.get("AKP_TEST_FASTEMBED"),
    reason="downloads a 65 MB model; set AKP_TEST_FASTEMBED=1",
)
def test_semantic_embeddings_beat_hashing_on_paraphrases(settings):
    paraphrases = load_retrieval_sets()["paraphrases"]
    scores = {}
    for provider in ("hash", "fastembed"):
        store = DocumentStore.from_settings(
            settings.model_copy(update={"embedding_provider": provider})
        )
        load_directory(store, CORPUS_DIR)
        scores[provider] = retrieval_metrics(store, paraphrases)["mrr"]
    assert scores["fastembed"] > scores["hash"] + 0.2
