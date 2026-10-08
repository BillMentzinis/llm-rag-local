"""
A deterministic stand-in for the sentence-transformers embedding model.

It hashes each word into a fixed-size count vector, so texts that share words
come out similar. Tests and the browser-test app use it to run offline.
"""

import hashlib
import re

import numpy as np


class FakeEmbedder:
    """Hashes each word into a fixed-size count vector; shared words mean similar texts."""

    dimension = 64

    def get_embedding_dimension(self):
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
