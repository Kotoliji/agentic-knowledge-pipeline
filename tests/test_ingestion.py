import asyncio
import os
from types import SimpleNamespace

import pytest

from akp.ingestion import (
    DEAD_LETTER_SUBJECT,
    IngestPublisher,
    IngestRequest,
    handle,
    run_worker,
)

MAX_DELIVER = 5


class FakeMsg:
    def __init__(self, data: bytes, delivered: int = 1):
        self.data = data
        self.metadata = SimpleNamespace(num_delivered=delivered)
        self.outcome = None

    async def ack(self):
        self.outcome = ("ack",)

    async def nak(self, delay=None):
        self.outcome = ("nak", delay)

    async def term(self):
        self.outcome = ("term",)


class FakeJetStream:
    def __init__(self):
        self.published = []

    async def publish(self, subject, payload, headers=None):
        self.published.append((subject, payload, headers))


class FakeStore:
    def __init__(self, error=None):
        self.error, self.stored = error, []

    def upsert_document(self, title, text):
        if self.error:
            raise self.error
        self.stored.append(title)
        return SimpleNamespace(title=title, doc_id="d1", created=True)


def doc(title="Parking", text="Permits are for level B2.") -> bytes:
    return IngestRequest(title=title, text=text).model_dump_json().encode()


async def test_stored_document_is_acked():
    msg, store, js = FakeMsg(doc()), FakeStore(), FakeJetStream()
    assert await handle(msg, store, js, MAX_DELIVER) == "acked"
    assert msg.outcome == ("ack",) and store.stored == ["Parking"]


async def test_transient_failure_is_retried_with_growing_delay():
    js = FakeJetStream()
    store = FakeStore(ConnectionError("retrieval service down"))
    first, third = FakeMsg(doc(), delivered=1), FakeMsg(doc(), delivered=3)
    assert await handle(first, store, js, MAX_DELIVER) == "retry"
    assert await handle(third, store, js, MAX_DELIVER) == "retry"
    assert first.outcome == ("nak", 2) and third.outcome == ("nak", 8)
    assert js.published == []


async def test_last_attempt_goes_to_the_dead_letter_subject():
    msg, js = FakeMsg(doc(), delivered=MAX_DELIVER), FakeJetStream()
    outcome = await handle(msg, FakeStore(ConnectionError("down")), js, MAX_DELIVER)
    assert outcome == "dead" and msg.outcome == ("term",)
    [(subject, payload, headers)] = js.published
    assert subject == DEAD_LETTER_SUBJECT and payload == msg.data
    assert "gave up after 5 attempts" in headers["Akp-Reason"]


@pytest.mark.parametrize(
    ("data", "store"),
    [
        (b"not json", FakeStore()),
        (b'{"title": "", "text": "x"}', FakeStore()),
        (doc(), FakeStore(ValueError("title and text are required"))),
    ],
)
async def test_invalid_or_rejected_documents_are_not_retried(data, store):
    msg, js = FakeMsg(data), FakeJetStream()
    assert await handle(msg, store, js, MAX_DELIVER) == "dead"
    assert msg.outcome == ("term",) and len(js.published) == 1


@pytest.mark.skipif(not os.environ.get("AKP_TEST_NATS_URL"), reason="set AKP_TEST_NATS_URL")
async def test_end_to_end_through_jetstream(store, settings):
    s = settings.model_copy(update={"nats_url": os.environ["AKP_TEST_NATS_URL"]})
    publisher = await IngestPublisher.connect(s)
    title = f"Parking {os.getpid()}-{id(store)}"
    request = IngestRequest(title=title, text="Each employee gets one permit for level B2.")
    first = await publisher.publish(request)
    again = await publisher.publish(request)
    await publisher.close()
    assert first.queued and not first.duplicate
    assert again.duplicate and again.doc_id == first.doc_id

    stop = asyncio.Event()
    worker = asyncio.create_task(run_worker(s, store, stop))
    for _ in range(100):
        if any(d.title == title for d in store.list_documents()):
            break
        await asyncio.sleep(0.1)
    stop.set()
    await asyncio.wait_for(worker, 10)
    assert [d.title for d in store.list_documents()].count(title) == 1
