import chromadb
import pytest
from chromadb.config import Settings

from conftest import make_chunks
from vector_store_manager import VectorStoreManager

CAT = "the cat sat on the mat"
FINANCE = "quarterly revenue grew twelve percent"


def test_collection_uses_cosine_space(store):
    assert store.collection.metadata["hnsw:space"] == "cosine"


def test_similarity_is_cosine(store):
    store.add_documents(make_chunks("a.txt", [CAT, FINANCE]))

    results = store.search(CAT, top_k=2, min_similarity=0.0)

    assert results[0]["text"] == CAT
    assert results[0]["similarity"] == pytest.approx(1.0, abs=1e-4)
    # No shared words means orthogonal vectors under the fake embedder
    assert results[1]["similarity"] == pytest.approx(0.0, abs=1e-4)


def test_min_similarity_drops_unrelated_chunks(store):
    store.add_documents(make_chunks("a.txt", [CAT, FINANCE]))

    results = store.search("cat on a mat", top_k=2, min_similarity=0.3)

    assert [r["text"] for r in results] == [CAT]


def test_filter_restricts_search_to_one_document(store):
    # The best global match lives in a.txt, but the filter must still find b.txt's chunk
    store.add_documents(make_chunks("a.txt", [CAT, "the cat again on the mat"]))
    store.add_documents(make_chunks("b.txt", ["a cat appears once", FINANCE]))

    results = store.search(CAT, top_k=1, min_similarity=0.0, filter_metadata={"filename": "b.txt"})

    assert [r["metadata"]["filename"] for r in results] == ["b.txt"]
    assert results[0]["text"] == "a cat appears once"


def test_reupload_replaces_and_drops_stale_chunks(store):
    store.add_documents(make_chunks("a.txt", ["one", "two", "three", "four"], content_hash="v1"))
    store.add_documents(make_chunks("a.txt", ["uno", "dos"], content_hash="v2"))

    stored = store.collection.get(where={"filename": "a.txt"}, include=["documents"])
    assert sorted(stored["documents"]) == ["dos", "uno"]
    assert store.get_document_metadata("a.txt")["content_hash"] == "v2"


def test_documents_with_different_names_do_not_collide(store):
    store.add_documents(make_chunks("a.txt", ["alpha"]))
    store.add_documents(make_chunks("b.txt", ["beta"]))

    assert store.collection.count() == 2
    assert {d["filename"] for d in store.list_documents()} == {"a.txt", "b.txt"}


def test_delete_and_clear(store):
    store.add_documents(make_chunks("a.txt", ["alpha", "more alpha"]))
    store.add_documents(make_chunks("b.txt", ["beta"]))

    assert store.delete_document("a.txt") == 2
    assert store.get_document_metadata("a.txt") is None
    assert store.clear_all() == 1
    assert store.is_empty()
    assert store.collection.metadata["hnsw:space"] == "cosine"


def _legacy_collection(path, name="documents"):
    """Create a collection the way older versions of the app did (L2 space, raw embeddings)."""
    client = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))
    return client, client.get_or_create_collection(name=name, metadata={"description": "Document chunks for RAG"})


def test_legacy_l2_index_is_migrated(tmp_path, embedder):
    path = str(tmp_path / "chroma")
    _, legacy = _legacy_collection(path)
    texts = [CAT, FINANCE]
    legacy.add(
        ids=["abcd1234_0", "abcd1234_1"],
        documents=texts,
        embeddings=embedder.encode(texts).tolist(),  # unnormalized, like the old code
        metadatas=[{"filename": "old.txt", "chunk_index": i} for i in range(2)],
    )

    store = VectorStoreManager(persist_directory=path, embedding_model=embedder)

    assert store.collection.metadata["hnsw:space"] == "cosine"
    assert store.collection.count() == 2
    results = store.search(CAT, top_k=1, min_similarity=0.0)
    assert results[0]["text"] == CAT
    assert results[0]["metadata"]["filename"] == "old.txt"
    assert results[0]["similarity"] == pytest.approx(1.0, abs=1e-4)
    assert [c.name for c in store.client.list_collections()] == ["documents"]


def test_migration_interrupted_after_delete_is_recovered(tmp_path, embedder):
    path = str(tmp_path / "chroma")
    client = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))
    migrated = client.create_collection("documents_cosine_migration", metadata={"hnsw:space": "cosine"})
    migrated.add(ids=["x_0"], documents=[CAT], embeddings=embedder.encode([CAT], normalize_embeddings=True).tolist(),
                 metadatas=[{"filename": "kept.txt"}])

    store = VectorStoreManager(persist_directory=path, embedding_model=embedder)

    assert [c.name for c in store.client.list_collections()] == ["documents"]
    assert store.list_documents()[0]["filename"] == "kept.txt"


def test_migration_interrupted_before_delete_restarts(tmp_path, embedder):
    path = str(tmp_path / "chroma")
    client, legacy = _legacy_collection(path)
    legacy.add(ids=["a_0"], documents=[CAT], embeddings=embedder.encode([CAT]).tolist(),
               metadatas=[{"filename": "old.txt"}])
    client.create_collection("documents_cosine_migration", metadata={"hnsw:space": "cosine"})

    store = VectorStoreManager(persist_directory=path, embedding_model=embedder)

    assert [c.name for c in store.client.list_collections()] == ["documents"]
    assert store.collection.metadata["hnsw:space"] == "cosine"
    assert store.collection.count() == 1
