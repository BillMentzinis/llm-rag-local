import pytest

from document_processor import DocumentProcessor
from rag_pipeline import RAGPipeline


@pytest.fixture
def pipeline(store):
    # Retrieval and ingestion don't touch the model, so none is loaded
    return RAGPipeline(model=None, tokenizer=None, vector_store=store,
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
