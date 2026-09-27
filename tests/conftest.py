"""
Shared test fixtures.

Tests run offline: a deterministic bag-of-words embedder stands in for the
sentence-transformers model, while ChromaDB runs for real in a temp directory.
"""

import os
import sys

import pytest

TESTS = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(TESTS), TESTS]  # the app's modules, and test helpers from any subfolder

from fake_embedder import FakeEmbedder  # noqa: E402,F401  (re-exported for the tests)
from vector_store_manager import VectorStoreManager  # noqa: E402


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def store(tmp_path, embedder):
    return VectorStoreManager(persist_directory=str(tmp_path / "chroma"), embedding_model=embedder)


def make_chunks(filename, texts, content_hash="hash", chunk_config="200/30"):
    """Build chunk dicts shaped like DocumentProcessor output."""
    return [
        {
            "text": text,
            "filename": filename,
            "file_type": "Text File",
            "chunk_index": i,
            "total_chunks": len(texts),
            "content_hash": content_hash,
            "chunk_config": chunk_config,
        }
        for i, text in enumerate(texts)
    ]
