import pytest

from document_processor import DocumentProcessor
from rag_pipeline import RAGPipeline, source_label


@pytest.fixture
def pipeline(store):
    # Retrieval and ingestion don't touch the model, so none is loaded
    return RAGPipeline(None, vector_store=store,
                       doc_processor=DocumentProcessor(chunk_size=16, chunk_overlap=2))


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_ingest_then_skip_unchanged(pipeline, tmp_path):
    path = _write(tmp_path, "notes.txt", "Cats are small furry animals.")

    first = pipeline.ingest_document(path)
    second = pipeline.ingest_document(path)

    assert first["success"] and not first["skipped"] and not first["replaced"]
    assert second["success"] and second["skipped"]
    assert pipeline.vector_store.collection.count() == first["chunks_added"]


def test_ingest_changed_file_replaces_it(pipeline, tmp_path):
    long_text = "\n\n".join(f"Paragraph {i} about cats and their many habits." for i in range(20))
    path = _write(tmp_path, "notes.txt", long_text)
    first = pipeline.ingest_document(path)

    _write(tmp_path, "notes.txt", "Now just one line about dogs.")
    second = pipeline.ingest_document(path)

    assert first["chunks_added"] > 1
    assert second["replaced"] and second["chunks_added"] == 1
    docs = pipeline.get_documents()
    assert [(d["filename"], d["chunk_count"]) for d in docs] == [("notes.txt", 1)]


def test_changed_chunk_settings_rechunk_an_unchanged_file(pipeline, store, tmp_path):
    long_text = "\n\n".join(f"Paragraph {i} about cats and their many habits." for i in range(20))
    path = _write(tmp_path, "notes.txt", long_text)
    first = pipeline.ingest_document(path)

    rechunking = RAGPipeline(None, vector_store=store,
                             doc_processor=DocumentProcessor(chunk_size=64, chunk_overlap=8))
    second = rechunking.ingest_document(path)

    assert not second["skipped"] and second["replaced"]
    assert second["chunks_added"] < first["chunks_added"]
    assert store.get_document_metadata("notes.txt")["chunk_config"] == "64/8"


def test_retrieve_context_honours_document_filter(pipeline, tmp_path):
    pipeline.ingest_document(_write(tmp_path, "cats.txt", "cats purr and chase mice"))
    pipeline.ingest_document(_write(tmp_path, "dogs.txt", "dogs bark and chase cats"))

    everywhere = pipeline.retrieve_context("cats chase", top_k=5, min_similarity=0.0)
    focused = pipeline.retrieve_context("cats chase", top_k=5, min_similarity=0.0,
                                        filter_metadata={"filename": "dogs.txt"})

    assert {c["metadata"]["filename"] for c in everywhere} == {"cats.txt", "dogs.txt"}
    assert {c["metadata"]["filename"] for c in focused} == {"dogs.txt"}


def test_ingest_failure_is_reported(pipeline, tmp_path):
    result = pipeline.ingest_document(str(tmp_path / "missing.txt"))
    assert result["success"] is False


def _write_pdf(tmp_path, name, pages):
    import pymupdf

    path = tmp_path / name
    pdf = pymupdf.open()
    for text in pages:
        pdf.new_page().insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()
    return str(path)


def test_retrieved_pdf_chunks_carry_their_pages(store, tmp_path):
    # No overlap, so the second chunk holds no text from page 1
    pipeline = RAGPipeline(None, vector_store=store, doc_processor=DocumentProcessor(chunk_size=16, chunk_overlap=0))
    path = _write_pdf(tmp_path, "animals.pdf", ["Cats purr when they are content and sleep in the sun.",
                                                   "Owls hunt quietly at night over the fields."])
    pipeline.ingest_document(path)

    [hit] = pipeline.retrieve_context("owls hunt at night", top_k=1)

    assert (hit["metadata"]["page_start"], hit["metadata"]["page_end"]) == (2, 2)
    assert source_label(hit["metadata"]) == "animals.pdf, p. 2"
    assert "[Source: animals.pdf, p. 2]" in pipeline.format_context_for_prompt([hit])


def test_pdfs_indexed_without_pages_are_indexed_again(pipeline, store, tmp_path):
    path = _write_pdf(tmp_path, "old.pdf", ["Cats purr when they are content."])
    pipeline.ingest_document(path)
    # Remove the pages, as an index made before they were recorded would have it
    # (update merges metadata, and None deletes a key)
    ids = store.collection.get(where={"filename": "old.pdf"})["ids"]
    store.collection.update(ids=ids, metadatas=[{"page_start": None, "page_end": None}] * len(ids))
    assert "page_start" not in store.get_document_metadata("old.pdf")

    again = pipeline.ingest_document(path)
    unchanged = pipeline.ingest_document(path)

    assert not again["skipped"] and again["replaced"]
    assert store.get_document_metadata("old.pdf")["page_start"] == 1
    assert unchanged["skipped"]


@pytest.mark.parametrize("metadata, label", [
    ({"filename": "report.pdf", "chunk_index": 4, "page_start": 3, "page_end": 3}, "report.pdf, p. 3"),
    ({"filename": "report.pdf", "chunk_index": 4, "page_start": 3, "page_end": 4}, "report.pdf, pp. 3-4"),
    ({"filename": "notes.txt", "chunk_index": 1}, "notes.txt, Chunk 2"),
    ({}, "unknown, Chunk 1"),
])
def test_source_labels(metadata, label):
    assert source_label(metadata) == label
