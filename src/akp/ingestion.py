"""Asynchronous document ingestion over NATS JetStream.

The API publishes documents and answers 202 at once; `python -m akp.ingest_worker`
consumes them and writes through the retrieval service. Delivery is at-least-once:

* a message is acked only after the document is stored;
* transient failures (retrieval service down) are nak'ed with a growing delay and
  redelivered; after the last allowed attempt the message goes to a dead-letter subject;
* invalid payloads and permanent errors go straight to the dead-letter subject;
* duplicates are dropped twice: JetStream discards a re-published message with the same
  Nats-Msg-Id (the content hash) within the dedup window, and a redelivered message is a
  no-op because the store upserts by content hash.
"""

import asyncio
import contextlib
import logging
import signal

import nats
from nats.errors import TimeoutError as NatsTimeout
from nats.js.api import ConsumerConfig, StreamConfig
from nats.js.errors import BadRequestError
from pydantic import BaseModel, Field, ValidationError

from akp.config import Settings, get_settings
from akp.metrics import INGEST_MESSAGES, INGEST_PENDING
from akp.retrieval import DocumentStore

log = logging.getLogger(__name__)

STREAM = "AKP_INGEST"
SUBJECT = "akp.ingest.documents"
DEAD_LETTER_SUBJECT = "akp.ingest.dead"
DURABLE = "ingest-worker"


class IngestRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=200_000)


class IngestAccepted(BaseModel):
    doc_id: str
    title: str
    queued: bool
    duplicate: bool


async def ensure_stream(js, settings: Settings) -> None:
    config = StreamConfig(
        name=STREAM,
        subjects=[SUBJECT, DEAD_LETTER_SUBJECT],
        duplicate_window=settings.ingest_dedup_window_s,
    )
    try:
        await js.add_stream(config)
    except BadRequestError:  # exists with an older config
        await js.update_stream(config)


class IngestPublisher:
    def __init__(self, nc, js):
        self._nc, self._js = nc, js

    @classmethod
    async def connect(cls, settings: Settings) -> "IngestPublisher":
        nc = await nats.connect(settings.nats_url, name="akp-api")
        js = nc.jetstream()
        await ensure_stream(js, settings)
        return cls(nc, js)

    async def publish(self, request: IngestRequest) -> IngestAccepted:
        doc_id = DocumentStore.doc_id_for(request.title, request.text)
        ack = await self._js.publish(
            SUBJECT, request.model_dump_json().encode(), headers={"Nats-Msg-Id": doc_id}
        )
        return IngestAccepted(
            doc_id=doc_id, title=request.title, queued=True, duplicate=bool(ack.duplicate)
        )

    async def close(self) -> None:
        await self._nc.drain()


async def dead_letter(js, msg, reason: str) -> None:
    await js.publish(DEAD_LETTER_SUBJECT, msg.data, headers={"Akp-Reason": reason[:500]})
    await msg.term()
    log.warning("dead-lettered message: %s", reason)


async def handle(msg, store, js, max_deliver: int) -> str:
    """Processes one message; returns "acked", "retry" or "dead"."""
    try:
        request = IngestRequest.model_validate_json(msg.data)
    except ValidationError as exc:
        await dead_letter(js, msg, f"invalid payload ({exc.error_count()} errors)")
        return "dead"
    try:
        result = await asyncio.to_thread(store.upsert_document, request.title, request.text)
    except (ConnectionError, TimeoutError) as exc:  # RetrievalUnavailable is a ConnectionError
        attempt = msg.metadata.num_delivered
        if attempt >= max_deliver:
            await dead_letter(js, msg, f"gave up after {attempt} attempts: {exc}")
            return "dead"
        await msg.nak(delay=min(2**attempt, 30))
        log.info("retrying %s (attempt %d): %s", request.title, attempt, exc)
        return "retry"
    except ValueError as exc:  # rejected by the service: retrying won't help
        await dead_letter(js, msg, f"rejected: {exc}")
        return "dead"
    await msg.ack()
    log.info("stored %s (%s, created=%s)", result.title, result.doc_id, result.created)
    return "acked"


async def run_worker(settings: Settings, store, stop: asyncio.Event) -> None:
    nc = await nats.connect(settings.nats_url, name="akp-ingest-worker")
    js = nc.jetstream()
    await ensure_stream(js, settings)
    sub = await js.pull_subscribe(
        SUBJECT,
        durable=DURABLE,
        config=ConsumerConfig(
            ack_wait=settings.ingest_ack_wait_s, max_deliver=settings.ingest_max_deliver
        ),
    )
    log.info("ingest worker consuming %s", SUBJECT)
    try:
        while not stop.is_set():
            try:
                messages = await sub.fetch(batch=10, timeout=1)
            except NatsTimeout:
                INGEST_PENDING.set((await sub.consumer_info()).num_pending)
                continue
            for msg in messages:
                outcome = await handle(msg, store, js, settings.ingest_max_deliver)
                INGEST_MESSAGES.labels(outcome).inc()
            INGEST_PENDING.set((await sub.consumer_info()).num_pending)
    finally:
        await nc.drain()


def main() -> None:
    from akp.retrieval_service import RetrievalClient

    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    store = RetrievalClient(settings.retrieval_grpc_target, settings.retrieval_timeout_s)

    async def run() -> None:
        stop = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):  # no signal handlers on Windows
                asyncio.get_running_loop().add_signal_handler(sig, stop.set)
        await run_worker(settings, store, stop)

    if settings.metrics_port:
        from prometheus_client import start_http_server

        start_http_server(settings.metrics_port)
    asyncio.run(run())
