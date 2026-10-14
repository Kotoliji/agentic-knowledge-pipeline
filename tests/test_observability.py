from types import SimpleNamespace

from akp.observability import Observability
from akp.prompt_sync import sync
from akp.prompts import PROMPTS, to_langchain

EXPECTED_VARIABLES = {
    "akp-router": {"question"},
    "akp-draft": {"question", "passages"},
    "akp-critic": {"question", "passages", "answer", "citations", "answerable"},
    "akp-chat": {"question"},
}


def test_prompt_variables_match_what_the_agents_pass():
    for name, variables in EXPECTED_VARIABLES.items():
        assert set(to_langchain(PROMPTS[name]).input_variables) == variables


class FakeLangfuse:
    def __init__(self, fallback=False, existing=None):
        self.fallback, self.existing = fallback, existing or {}
        self.fetches, self.created = 0, []

    def get_prompt(self, name, **kwargs):
        self.fetches += 1
        if kwargs.get("fallback") is None and name not in self.existing:
            raise LookupError(name)
        messages = self.existing.get(name, kwargs.get("fallback"))
        return SimpleNamespace(
            is_fallback=self.fallback,
            prompt=messages,
            get_langchain_prompt=lambda: [
                (m["role"], m["content"].replace("{{", "{").replace("}}", "}")) for m in messages
            ],
        )

    def create_prompt(self, **kwargs):
        self.created.append(kwargs["name"])


def test_disabled_observability_uses_local_prompts_and_no_callbacks():
    obs = Observability()
    assert not obs.enabled
    assert obs.new_trace() == (None, [])
    assert obs.prompt("akp-router").input_variables == ["question"]


def test_managed_prompt_is_linked_to_generations():
    obs = Observability(FakeLangfuse())
    template = obs.prompt("akp-draft")
    assert "langfuse_prompt" in template.metadata
    assert set(template.input_variables) == {"question", "passages"}


def test_unreachable_langfuse_falls_back_once_and_stops_fetching():
    client = FakeLangfuse(fallback=True)
    obs = Observability(client)
    first, second = obs.prompt("akp-router"), obs.prompt("akp-critic")
    assert client.fetches == 1
    assert not first.metadata and not second.metadata


def test_sync_publishes_only_changed_prompts():
    unchanged = {"akp-router": PROMPTS["akp-router"]}
    client = FakeLangfuse(existing=unchanged)
    assert sorted(sync(client, "production")) == ["akp-chat", "akp-critic", "akp-draft"]
