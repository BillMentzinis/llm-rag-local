"""
Shared test fixtures.

Tests run offline: a deterministic bag-of-words embedder stands in for the
sentence-transformers model, while ChromaDB runs for real in a temp directory.
"""

import hashlib
import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store_manager import VectorStoreManager  # noqa: E402


class FakeEmbedder:
    """Hashes each word into a fixed-size count vector; shared words mean similar texts."""

    dimension = 64

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, show_progress_bar=False, normalize_embeddings=False):
        vectors = np.zeros((len(texts), self.dimension))
        for row, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension
                vectors[row, bucket] += 1.0
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.where(norms == 0, 1, norms)
        return vectors


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def store(tmp_path, embedder):
    return VectorStoreManager(persist_directory=str(tmp_path / "chroma"), embedding_model=embedder)


def make_chunks(filename, texts, content_hash="hash"):
    """Build chunk dicts shaped like DocumentProcessor output."""
    return [
        {
            "text": text,
            "filename": filename,
            "file_type": "Text File",
            "chunk_index": i,
            "total_chunks": len(texts),
            "content_hash": content_hash,
        }
        for i, text in enumerate(texts)
    ]
