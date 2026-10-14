"""Publish the prompts in akp/prompts.py to Langfuse.

A new version is created only when a prompt's text changed, and it gets the
configured label (default "production"), which is what the running service
fetches. Usage: python -m akp.prompt_sync
"""

import sys

from akp.config import get_settings
from akp.observability import Observability
from akp.prompts import PROMPTS


def _normalise(messages) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def sync(client, label: str) -> list[str]:
    published = []
    for name, messages in PROMPTS.items():
        try:
            current = client.get_prompt(
                name, type="chat", label=label, cache_ttl_seconds=0, max_retries=0
            )
            if _normalise(current.prompt) == messages:
                continue
        except Exception:  # noqa: BLE001 - not found yet: create the first version
            pass
        client.create_prompt(
            name=name,
            prompt=messages,
            type="chat",
            labels=[label],
            commit_message="Synced from akp/prompts.py",
        )
        published.append(name)
    return published


def main() -> None:
    settings = get_settings()
    obs = Observability.from_settings(settings)
    if not obs.enabled:
        sys.exit("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set")
    published = sync(obs.client, settings.langfuse_prompt_label)
    print("published:", ", ".join(published) if published else "nothing changed")
    obs.flush()


if __name__ == "__main__":
    main()
