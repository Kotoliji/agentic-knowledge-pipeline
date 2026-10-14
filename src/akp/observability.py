import logging

from langchain_core.prompts import ChatPromptTemplate

from akp.config import Settings
from akp.prompts import PROMPTS, to_langchain

log = logging.getLogger(__name__)


class Observability:
    """Langfuse tracing and prompt management, or local no-ops when not configured."""

    def __init__(self, client=None, prompt_label: str = "production"):
        self.client = client
        self.prompt_label = prompt_label
        self._prompts_unreachable = False

    @classmethod
    def from_settings(cls, settings: Settings) -> "Observability":
        if not (settings.langfuse_public_key and settings.langfuse_secret_key):
            return cls()
        from langfuse import Langfuse

        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            base_url=settings.langfuse_base_url,
            environment=settings.environment,
        )
        return cls(client, settings.langfuse_prompt_label)

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def prompt(self, name: str) -> ChatPromptTemplate:
        local = PROMPTS[name]
        if not self.client or self._prompts_unreachable:
            return to_langchain(local)
        managed = self.client.get_prompt(
            name,
            type="chat",
            label=self.prompt_label,
            fallback=local,
            max_retries=1,
            fetch_timeout_seconds=3,
        )
        if managed.is_fallback:
            # Don't pay a fetch timeout on every request while Langfuse is down.
            log.warning("Langfuse prompt %r unavailable; using local prompts", name)
            self._prompts_unreachable = True
            return to_langchain(local)
        template = ChatPromptTemplate.from_messages(managed.get_langchain_prompt())
        # Links each generation in the trace to the exact prompt version used.
        template.metadata = {"langfuse_prompt": managed}
        return template

    def new_trace(self) -> tuple[str | None, list]:
        """Returns (trace_id, callbacks) for one graph run."""
        if not self.client:
            return None, []
        from langfuse.langchain import CallbackHandler

        trace_id = self.client.create_trace_id()
        return trace_id, [CallbackHandler(trace_context={"trace_id": trace_id})]

    def trace_url(self, trace_id: str | None) -> str | None:
        if not (self.client and trace_id):
            return None
        return self.client.get_trace_url(trace_id=trace_id)

    def score(self, trace_id: str | None, name: str, value: bool, comment: str = "") -> None:
        if self.client and trace_id:
            self.client.create_score(
                trace_id=trace_id,
                name=name,
                value=1.0 if value else 0.0,
                data_type="BOOLEAN",
                comment=comment or None,
            )

    def flush(self) -> None:
        if self.client:
            self.client.flush()
