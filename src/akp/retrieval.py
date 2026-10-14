import asyncio
import hashlib
import math
import re
import uuid
from pathlib import Path
from typing import Any, Protocol

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel
from qdrant_client import QdrantClient, models

from akp.config import Settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i if in is it its of on or "
    "our should so that the their there this to was we what when where which who why will "
    "with you your my me us they them than then into about after before any all".split()
)
_POINT_NAMESPACE = uuid.UUID("6f1c2d3e-8a4b-4c5d-9e0f-112233445566")


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _stem(token: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


class HashEmbedder:
    """Feature-hashing bag-of-words vectors.

    Lexical rather than semantic, but deterministic and dependency-free, so the
    pipeline (and CI) runs without downloading a model or calling an API.
    """

    def __init__(self, dim: int):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            vec[int.from_bytes(digest, "big") % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text.strip()):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            units.append(para)
        else:
            units.extend(s for s in re.split(r"(?<=[.!?])\s+", para) if s)

    chunks: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + 1 + len(unit) > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = tail[tail.find(" ") + 1 :] if " " in tail else ""
        current = f"{current}\n{unit}" if current else unit
    if current:
        chunks.append(current)
    return chunks


class Passage(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float = 0.0


class DocumentInfo(BaseModel):
    doc_id: str
    title: str
    chunks: int


class IngestResult(BaseModel):
    doc_id: str
    title: str
    chunks: int
    created: bool


class SearchBackend(Protocol):
    def search(self, query: str, k: int) -> list[Passage]: ...


class DocumentStore:
    def __init__(self, client: QdrantClient, settings: Settings):
        self.client = client
        self.settings = settings
        self.collection = settings.collection
        self.embedder = HashEmbedder(settings.embedding_dim)
        self.ensure_collection()

    @classmethod
    def from_settings(cls, settings: Settings) -> "DocumentStore":
        url = settings.qdrant_url
        client = QdrantClient(location=url) if url == ":memory:" else QdrantClient(url=url)
        return cls(client, settings)

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(
                size=self.settings.embedding_dim, distance=models.Distance.COSINE
            ),
        )

    @staticmethod
    def doc_id_for(title: str, text: str) -> str:
        return hashlib.sha256(f"{title}\n{text}".encode()).hexdigest()[:16]

    def upsert_document(self, title: str, text: str) -> IngestResult:
        # The id is a content hash, so re-ingesting the same document is a no-op —
        # duplicate deliveries from the ingestion queue are safe by construction.
        doc_id = self.doc_id_for(title, text)
        existing = self._chunk_count(doc_id)
        if existing:
            return IngestResult(doc_id=doc_id, title=title, chunks=existing, created=False)

        chunks = chunk_text(text, self.settings.chunk_size, self.settings.chunk_overlap)
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(_POINT_NAMESPACE, f"{doc_id}:{i}")),
                vector=self.embedder.embed(f"{title}\n{chunk}"),
                payload={"doc_id": doc_id, "title": title, "text": chunk, "index": i},
            )
            for i, chunk in enumerate(chunks)
        ]
        if points:
            self.client.upsert(self.collection, points=points)
        return IngestResult(doc_id=doc_id, title=title, chunks=len(points), created=True)

    def search(self, query: str, k: int) -> list[Passage]:
        hits = self.client.query_points(
            self.collection, query=self.embedder.embed(query), limit=k, with_payload=True
        ).points
        return [
            Passage(
                chunk_id=f"{h.payload['doc_id']}:{h.payload['index']}",
                doc_id=h.payload["doc_id"],
                title=h.payload["title"],
                text=h.payload["text"],
                score=h.score,
            )
            for h in hits
        ]

    def list_documents(self) -> list[DocumentInfo]:
        docs: dict[str, DocumentInfo] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=256, offset=offset, with_payload=["doc_id", "title"]
            )
            for p in points:
                info = docs.setdefault(
                    p.payload["doc_id"],
                    DocumentInfo(doc_id=p.payload["doc_id"], title=p.payload["title"], chunks=0),
                )
                info.chunks += 1
            if offset is None:
                return sorted(docs.values(), key=lambda d: d.title)

    def delete_document(self, doc_id: str) -> bool:
        if not self._chunk_count(doc_id):
            return False
        self.client.delete(
            self.collection, points_selector=models.FilterSelector(filter=self._doc_filter(doc_id))
        )
        return True

    def _chunk_count(self, doc_id: str) -> int:
        return self.client.count(
            self.collection, count_filter=self._doc_filter(doc_id), exact=True
        ).count

    @staticmethod
    def _doc_filter(doc_id: str) -> models.Filter:
        return models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        )


class PassageRetriever(BaseRetriever):
    """LangChain retriever over any search backend (local store or remote service)."""

    backend: Any  # SearchBackend; a Protocol can't be a pydantic field type
    k: int = 4

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return [_to_document(p) for p in self.backend.search(query, self.k)]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        passages = await asyncio.to_thread(self.backend.search, query, self.k)
        return [_to_document(p) for p in passages]


def _to_document(p: Passage) -> Document:
    return Document(page_content=p.text, metadata=p.model_dump(exclude={"text"}))


def passage_from_document(doc: Document) -> Passage:
    return Passage(text=doc.page_content, **doc.metadata)


def load_directory(store: DocumentStore, directory: str | Path) -> list[IngestResult]:
    results = []
    for path in sorted(Path(directory).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        results.append(store.upsert_document(_title_of(text, path), text))
    return results


def _title_of(text: str, path: Path) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem.replace("-", " ").title()
