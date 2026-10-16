"""Evaluate the agent graph with an LLM judge, and the judge against human labels.

    python -m akp.evaluation run [--limit N]          # answer + judge the dataset
    python -m akp.evaluation label RUN.jsonl          # label the same answers yourself
    python -m akp.evaluation agreement RUN.jsonl      # judge vs human: agreement, kappa
    python -m akp.evaluation retrieval                # embedders compared, no LLM calls

Human labelling never shows the judge's verdict, so it can't anchor the labeller.
"""

import argparse
import asyncio
import json
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from akp.agents import chat_model, format_passages, make_agents
from akp.config import Settings, get_settings
from akp.graph import build_graph, passages_of
from akp.observability import Observability
from akp.retrieval import DocumentStore, Passage, PassageRetriever, load_directory, tokenize

EVAL_DIR = Path("evals")
DATASET = EVAL_DIR / "dataset.jsonl"
PARAPHRASES = EVAL_DIR / "retrieval_paraphrases.jsonl"
RUNS_DIR = EVAL_DIR / "runs"
LABELS_DIR = EVAL_DIR / "labels"
CRITERIA = ("correct", "faithful")


class EvalItem(BaseModel):
    id: str
    question: str
    reference: str
    answerable: bool
    source: str | None = None


class RetrievalItem(BaseModel):
    id: str
    question: str
    source: str


class Verdict(BaseModel):
    reasoning: str
    correct: bool
    faithful: bool


class EvalRecord(BaseModel):
    item: EvalItem
    answer: str
    status: str
    citations: list[Passage]
    search_queries: list[str]
    trace_id: str | None
    verdict: Verdict | None = None
    error: str | None = None


class HumanLabel(BaseModel):
    id: str
    correct: bool
    faithful: bool


class Judge(Protocol):
    async def grade(
        self, item: EvalItem, answer: str, citations: list[Passage], config: dict
    ) -> Verdict: ...


class LLMJudge:
    def __init__(self, settings: Settings, observability: Observability):
        self._obs = observability
        self._timeout_s = settings.llm_timeout_s
        model = chat_model(settings, "judge", 16000)
        self._model = model.with_structured_output(Verdict, method="json_schema")

    async def grade(self, item, answer, citations, config) -> Verdict:
        chain = self._obs.prompt("akp-judge") | self._model
        inputs = {
            "question": item.question,
            "answerable": "yes" if item.answerable else "no",
            "reference": item.reference,
            "answer": answer,
            "passages": format_passages(citations),
        }
        return await asyncio.wait_for(
            chain.ainvoke(inputs, {**config, "run_name": "judge"}), self._timeout_s
        )


class HeuristicJudge:
    """Offline stand-in for CI: numbers/keyword overlap. Not a real judge."""

    async def grade(self, item, answer, citations, config) -> Verdict:
        declined = not citations
        if not item.answerable:
            correct = declined
        elif declined:
            correct = False
        else:
            numbers = set(re.findall(r"\d+", item.reference))
            if numbers:
                correct = numbers <= set(re.findall(r"\d+", answer))
            else:
                ref = set(tokenize(item.reference))
                correct = len(ref & set(tokenize(answer))) >= len(ref) / 2
        faithful = declined or any(answer.strip() in c.text for c in citations)
        return Verdict(reasoning="heuristic", correct=correct, faithful=faithful)


def load_items(path: Path = DATASET) -> list[EvalItem]:
    return [EvalItem.model_validate_json(line) for line in path.read_text().splitlines() if line]


async def run_eval(
    items: list[EvalItem],
    graph,
    judge: Judge,
    obs: Observability,
    run_name: str,
    concurrency: int = 4,
    on_record: Callable[[EvalRecord], None] | None = None,
    max_consecutive_errors: int = 3,
) -> list[EvalRecord]:
    """Evaluates items concurrently. A failing item is recorded with its error instead
    of aborting the run; after several failures in a row (typically an exhausted API
    quota) the remaining items are skipped, so a later --resume picks them up."""
    semaphore = asyncio.Semaphore(concurrency)
    consecutive_errors = 0

    async def evaluate(item: EvalItem) -> EvalRecord | None:
        nonlocal consecutive_errors
        async with semaphore:
            if consecutive_errors >= max_consecutive_errors:
                return None
            trace_id, callbacks = obs.new_trace()
            config = {"callbacks": callbacks}
            try:
                with obs.attributes(session_id=run_name, tags=["eval"]):
                    state = await graph.ainvoke(
                        {"question": item.question}, {**config, "run_name": "ask"}
                    )
                    citations = passages_of(state, "citations")
                    verdict = await judge.grade(item, state["answer"], citations, config)
            except Exception as exc:  # noqa: BLE001 - recorded per item, run continues
                consecutive_errors += 1
                record = EvalRecord(
                    item=item,
                    answer="",
                    status="error",
                    citations=[],
                    search_queries=[],
                    trace_id=trace_id,
                    error=f"{type(exc).__name__}: {str(exc)[:300]}",
                )
            else:
                consecutive_errors = 0
                for criterion in CRITERIA:
                    obs.score(
                        trace_id,
                        f"judge_{criterion}",
                        getattr(verdict, criterion),
                        verdict.reasoning,
                    )
                record = EvalRecord(
                    item=item,
                    answer=state["answer"],
                    status=state["status"],
                    citations=citations,
                    search_queries=state.get("tried_queries", []),
                    trace_id=trace_id,
                    verdict=verdict,
                )
            if on_record:
                on_record(record)
            return record

    results = await asyncio.gather(*(evaluate(i) for i in items))
    return [r for r in results if r is not None]


def summarize(records: list[EvalRecord]) -> dict:
    def rate(rs: list[EvalRecord], criterion: str) -> float | None:
        return round(sum(getattr(r.verdict, criterion) for r in rs) / len(rs), 3) if rs else None

    errors = [r for r in records if r.verdict is None]
    records = [r for r in records if r.verdict is not None]
    answerable = [r for r in records if r.item.answerable]
    unanswerable = [r for r in records if not r.item.answerable]
    statuses: dict[str, int] = {}
    for r in records:
        statuses[r.status] = statuses.get(r.status, 0) + 1
    return {
        "graded_items": len(records),
        "errored_items": len(errors),
        "correct": rate(records, "correct"),
        "faithful": rate(records, "faithful"),
        "correct_on_answerable": rate(answerable, "correct"),
        "correct_decline_on_unanswerable": rate(unanswerable, "correct"),
        "needed_second_retrieval": sum(len(r.search_queries) > 1 for r in records),
        "statuses": statuses,
    }


def load_retrieval_sets() -> dict[str, list[RetrievalItem]]:
    """The answerable dataset questions (worded close to the documents) and a set of
    paraphrases that deliberately avoid the documents' vocabulary."""
    dataset = [
        RetrievalItem(id=i.id, question=i.question, source=i.source)
        for i in load_items()
        if i.answerable and i.source
    ]
    paraphrases = [
        RetrievalItem.model_validate_json(line)
        for line in PARAPHRASES.read_text().splitlines()
        if line
    ]
    return {"dataset": dataset, "paraphrases": paraphrases}


def retrieval_metrics(store, items: list[RetrievalItem], k: int = 4, depth: int = 10) -> dict:
    """Document-level: rank of the first retrieved chunk that comes from the source doc."""
    ranks: dict[str, int | None] = {}
    for item in items:
        titles = [p.title for p in store.search(item.question, depth)]
        ranks[item.id] = titles.index(item.source) + 1 if item.source in titles else None
    n = len(ranks)
    return {
        "questions": n,
        "recall@1": round(sum(r == 1 for r in ranks.values()) / n, 3),
        f"recall@{k}": round(sum(r is not None and r <= k for r in ranks.values()) / n, 3),
        "mrr": round(sum(1 / r for r in ranks.values() if r) / n, 3),
        "not_first": sorted(i for i, r in ranks.items() if r != 1),
    }


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """None when undefined: if both raters gave one and the same label to every item,
    chance agreement is 1 and kappa is 0/0, so it says nothing about the judge."""
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    p_a = sum(a for a, _ in pairs) / n
    p_b = sum(b for _, b in pairs) / n
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


def agreement(records: list[EvalRecord], labels: list[HumanLabel]) -> dict:
    by_id = {label.id: label for label in labels}
    labelled = [r for r in records if r.item.id in by_id and r.verdict is not None]
    report: dict = {"labelled_items": len(labelled)}
    for criterion in CRITERIA:
        pairs = [
            (getattr(r.verdict, criterion), getattr(by_id[r.item.id], criterion)) for r in labelled
        ]
        if not pairs:
            continue
        report[criterion] = {
            "agreement": round(sum(a == b for a, b in pairs) / len(pairs), 3),
            "cohen_kappa": None if (kappa := cohen_kappa(pairs)) is None else round(kappa, 3),
            "human_no": sum(not b for _, b in pairs),
            "judge_yes_human_no": sum(a and not b for a, b in pairs),
            "judge_no_human_yes": sum(b and not a for a, b in pairs),
            "disagreements": [
                r.item.id for r, (a, b) in zip(labelled, pairs, strict=True) if a != b
            ],
        }
    return report


def read_records(path: Path) -> list[EvalRecord]:
    """Records are appended as they finish; for a resumed item the latest one wins."""
    latest: dict[str, EvalRecord] = {}
    for line in path.read_text().splitlines():
        if line:
            record = EvalRecord.model_validate_json(line)
            latest[record.item.id] = record
    return list(latest.values())


def labels_path_for(run_path: Path) -> Path:
    return LABELS_DIR / run_path.name


def read_labels(path: Path) -> list[HumanLabel]:
    if not path.exists():
        return []
    return [HumanLabel.model_validate_json(line) for line in path.read_text().splitlines() if line]


def build_eval_graph(settings: Settings, obs: Observability):
    store = DocumentStore.from_settings(settings.model_copy(update={"qdrant_url": ":memory:"}))
    load_directory(store, "corpus")
    retriever = PassageRetriever(backend=store, k=settings.top_k)
    return build_graph(make_agents(settings, obs), retriever, settings)


async def _run(args: argparse.Namespace) -> None:
    settings = get_settings()
    obs = Observability.from_settings(settings)
    graph = build_eval_graph(settings, obs)
    real = settings.llm_provider != "offline"
    judge: Judge = LLMJudge(settings, obs) if real else HeuristicJudge()
    if args.resume:
        run_path = Path(args.resume)
        done = {r.item.id for r in read_records(run_path) if r.verdict is not None}
    else:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        run_path = RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{settings.llm_provider}.jsonl"
        done = set()
    items = [i for i in load_items() if i.id not in done][: args.limit]
    print(f"{len(done)} already graded, running {len(items)}")

    with run_path.open("a") as out:

        def persist(record: EvalRecord) -> None:
            out.write(record.model_dump_json() + "\n")
            out.flush()

        await run_eval(items, graph, judge, obs, run_path.stem, args.concurrency, persist)
    obs.flush()

    summary = summarize(read_records(run_path))
    run_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nrun written to {run_path}")


def _retrieval(args: argparse.Namespace) -> None:
    settings = get_settings()
    sets = load_retrieval_sets()
    report = {}
    for provider in ("hash", "fastembed"):
        store_settings = settings.model_copy(
            update={"embedding_provider": provider, "qdrant_url": ":memory:"}
        )
        store = DocumentStore.from_settings(store_settings)
        load_directory(store, "corpus")
        name = store.embedder.name
        report[name] = {
            set_name: retrieval_metrics(store, items, k=settings.top_k)
            for set_name, items in sets.items()
        }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f"retrieval-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwritten to {out}")


def _ask(prompt: str) -> bool | None:
    while True:
        answer = input(prompt).strip().lower()
        if answer in ("y", "n"):
            return answer == "y"
        if answer == "q":
            return None


def _label(args: argparse.Namespace) -> None:
    run_path = Path(args.run)
    out = labels_path_for(run_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {label.id for label in read_labels(out)}
    obs = Observability.from_settings(get_settings())
    todo = [r for r in read_records(run_path) if r.verdict and r.item.id not in done]
    print(f"{len(done)} labelled, {len(todo)} to go. Answer y/n, q to stop.\n")
    for r in todo:
        print("=" * 80)
        print(f"[{r.item.id}] {r.item.question}")
        print(f"Answerable from docs: {r.item.answerable} | Reference: {r.item.reference}")
        print(f"\nAnswer ({r.status}): {r.answer}")
        for i, c in enumerate(r.citations, 1):
            print(f"\n  [{i}] {c.title}: {c.text}")
        correct = _ask("\ncorrect? ")
        if correct is None:
            break
        faithful = _ask("faithful (every claim backed by the cited passages)? ")
        if faithful is None:
            break
        label = HumanLabel(id=r.item.id, correct=correct, faithful=faithful)
        with out.open("a") as f:
            f.write(label.model_dump_json() + "\n")
        obs.score(r.trace_id, "human_correct", correct)
        obs.score(r.trace_id, "human_faithful", faithful)
    obs.flush()
    print(f"\nlabels saved to {out}")


def _agreement(args: argparse.Namespace) -> None:
    run_path = Path(args.run)
    report = agreement(read_records(run_path), read_labels(labels_path_for(run_path)))
    run_path.with_suffix(".agreement.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m akp.evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="answer and judge the dataset")
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--resume", metavar="RUN.jsonl", help="continue a run, retrying failures")
    sub.add_parser("label", help="label a run's answers by hand").add_argument("run")
    sub.add_parser("agreement", help="judge vs human agreement").add_argument("run")
    sub.add_parser("retrieval", help="compare embedders on retrieval metrics (no LLM)")
    args = parser.parse_args()
    if args.command == "run":
        asyncio.run(_run(args))
    elif args.command == "label":
        _label(args)
    elif args.command == "retrieval":
        _retrieval(args)
    else:
        _agreement(args)


if __name__ == "__main__":
    main()
