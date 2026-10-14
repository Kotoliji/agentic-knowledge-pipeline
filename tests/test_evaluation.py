import pytest

from akp.agents import OfflineAgents
from akp.evaluation import (
    EvalItem,
    EvalRecord,
    HeuristicJudge,
    HumanLabel,
    LLMJudge,
    Verdict,
    agreement,
    cohen_kappa,
    load_items,
    run_eval,
    summarize,
)
from akp.graph import build_graph
from akp.observability import Observability
from akp.retrieval import PassageRetriever
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

    assert summary["items"] == 30
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
        ([(True, True)] * 10, 1.0),
        # 20 items: both say yes 14 times, both no 2, disagree 4 -> po=0.8, pe=0.68
        (
            [(True, True)] * 14 + [(False, False)] * 2 + [(True, False)] * 2 + [(False, True)] * 2,
            (0.8 - 0.68) / (1 - 0.68),
        ),
    ],
)
def test_cohen_kappa(pairs, expected):
    assert cohen_kappa(pairs) == pytest.approx(expected)


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
