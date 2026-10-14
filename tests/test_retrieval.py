import pytest

from akp.retrieval import HashEmbedder, PassageRetriever, chunk_text, passage_from_document


def test_embedder_is_deterministic_and_normalised():
    emb = HashEmbedder(256)
    a, b = emb.embed("Expense claims need receipts"), emb.embed("Expense claims need receipts")
    assert a == b
    assert sum(v * v for v in a) == pytest.approx(1.0)


def test_empty_text_embeds_to_zero_vector_without_error():
    assert set(HashEmbedder(64).embed("the and of")) == {0.0}


def test_chunks_respect_size_and_keep_all_paragraphs():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 30 for i in range(10))
    chunks = chunk_text(text, size=400, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 400 + 200 for c in chunks)
    for i in range(10):
        assert any(f"Paragraph {i}" in c for c in chunks)


def test_oversized_paragraph_is_split_on_sentences():
    para = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = chunk_text(para, size=300, overlap=0)
    assert len(chunks) > 1
    assert all(c.endswith(".") for c in chunks)


def test_reingesting_same_document_is_a_noop(store):
    first = store.upsert_document("Policy", "Claims must be filed within 60 days.")
    second = store.upsert_document("Policy", "Claims must be filed within 60 days.")
    assert first.created and not second.created
    assert first.doc_id == second.doc_id
    assert len(store.list_documents()) == 1


def test_search_finds_the_relevant_document(loaded_store):
    passages = loaded_store.search("hotel limit per night in London", k=3)
    assert passages[0].title == "Expense Policy"
    assert "220 EUR" in passages[0].text


def test_delete_document(loaded_store):
    doc = next(d for d in loaded_store.list_documents() if d.title == "Remote Work")
    assert loaded_store.delete_document(doc.doc_id)
    assert not loaded_store.delete_document(doc.doc_id)
    assert all(d.title != "Remote Work" for d in loaded_store.list_documents())


async def test_langchain_retriever_round_trips_passages(loaded_store):
    retriever = PassageRetriever(backend=loaded_store, k=2)
    docs = await retriever.ainvoke("How long is the on-call rotation?")
    passages = [passage_from_document(d) for d in docs]
    assert len(passages) == 2
    assert passages[0].title == "On-Call Rotation"
    assert passages[0].chunk_id.startswith(passages[0].doc_id)
