import re
from typing import Literal, Protocol

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from akp import prompts
from akp.config import Settings
from akp.retrieval import Passage, tokenize


class RouteDecision(BaseModel):
    route: Literal["retrieve", "direct"]
    search_query: str = Field(description="Standalone search query; empty for 'direct'")


class Draft(BaseModel):
    answer: str
    citations: list[int] = Field(description="Numbers of the passages that support the answer")
    answerable: bool


class Critique(BaseModel):
    supported: bool
    issues: str
    better_query: str = Field(description="Different search query to try; empty if none")


class Agents(Protocol):
    async def route(self, question: str) -> RouteDecision: ...
    async def draft(self, question: str, passages: list[Passage]) -> Draft: ...
    async def critique(self, question: str, draft: Draft, passages: list[Passage]) -> Critique: ...
    async def chat(self, question: str) -> str: ...


def format_passages(passages: list[Passage]) -> str:
    if not passages:
        return "(no passages found)"
    return "\n\n".join(f"[{i}] ({p.title})\n{p.text}" for i, p in enumerate(passages, 1))


# Server-side refusal fallback is only accepted on these models.
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1"}


def _chat_model(model: str, effort: str, max_tokens: int) -> ChatAnthropic:
    extra = {}
    if model in _FALLBACK_MODELS:
        extra = {
            "betas": ["server-side-fallback-2026-07-01"],
            "model_kwargs": {"fallbacks": "default"},
        }
    return ChatAnthropic(
        model=model,
        max_tokens=max_tokens,
        output_config={"effort": effort},
        max_retries=2,
        default_request_timeout=120,
        **extra,
    )


class ClaudeAgents:
    def __init__(self, settings: Settings):
        router = _chat_model(settings.model_router, settings.effort_router, 8000)
        answer = _chat_model(settings.model_answer, settings.effort_answer, 16000)
        critic = _chat_model(settings.model_critic, settings.effort_critic, 8000)
        # json_schema = native structured outputs. The default ("function_calling")
        # forces tool_choice, which current models reject.
        self._route = prompts.ROUTER | router.with_structured_output(
            RouteDecision, method="json_schema"
        )
        self._draft = prompts.DRAFT | answer.with_structured_output(Draft, method="json_schema")
        self._critique = prompts.CRITIC | critic.with_structured_output(
            Critique, method="json_schema"
        )
        self._chat = prompts.CHAT | router | StrOutputParser()

    async def route(self, question: str) -> RouteDecision:
        return await self._route.ainvoke({"question": question}, {"run_name": "router"})

    async def draft(self, question: str, passages: list[Passage]) -> Draft:
        return await self._draft.ainvoke(
            {"question": question, "passages": format_passages(passages)}, {"run_name": "drafter"}
        )

    async def critique(self, question: str, draft: Draft, passages: list[Passage]) -> Critique:
        return await self._critique.ainvoke(
            {
                "question": question,
                "passages": format_passages(passages),
                "answer": draft.answer,
                "citations": draft.citations,
                "answerable": draft.answerable,
            },
            {"run_name": "critic"},
        )

    async def chat(self, question: str) -> str:
        return await self._chat.ainvoke({"question": question}, {"run_name": "chat"})


_SMALL_TALK = {"hi", "hello", "hey", "thanks", "thank", "help", "bye"}


class OfflineAgents:
    """Deterministic keyword-overlap agents: no API key, used by tests, CI and demos.

    They make the graph's control flow observable without an LLM; answer quality is
    obviously far below the real agents.
    """

    min_overlap = 0.5

    async def route(self, question: str) -> RouteDecision:
        words = set(re.findall(r"[a-z]+", question.lower()))
        if words and words <= _SMALL_TALK | {"you", "can", "what", "do", "there"}:
            return RouteDecision(route="direct", search_query="")
        return RouteDecision(route="retrieve", search_query=question)

    async def draft(self, question: str, passages: list[Passage]) -> Draft:
        q = set(tokenize(question))
        best_score, best_idx, best_sentence = 0.0, 0, ""
        for i, p in enumerate(passages, 1):
            for sentence in re.split(r"(?<=[.!?])\s+|\n", p.text):
                score = len(q & set(tokenize(sentence))) / (len(q) or 1)
                if score > best_score:
                    best_score, best_idx, best_sentence = score, i, sentence.strip()
        if best_score < self.min_overlap:
            return Draft(answer="", citations=[], answerable=False)
        return Draft(answer=best_sentence, citations=[best_idx], answerable=True)

    async def critique(self, question: str, draft: Draft, passages: list[Passage]) -> Critique:
        cited = [passages[i - 1].text for i in draft.citations if 0 < i <= len(passages)]
        supported = draft.answerable and any(draft.answer in text for text in cited)
        return Critique(
            supported=supported,
            issues="" if supported else "No passage states an answer to the question.",
            better_query="",
        )

    async def chat(self, question: str) -> str:
        return "Hi! Ask me about company policies and processes — I answer with citations."


def make_agents(settings: Settings) -> Agents:
    if settings.llm_provider == "anthropic":
        return ClaudeAgents(settings)
    return OfflineAgents()
