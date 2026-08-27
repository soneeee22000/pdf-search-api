"""Test fixtures.

Everything here runs without network access and without downloading a model:
the embedder is faked and the token counter is injected. The real tokenizer is
exercised once, manually, in the ingestion smoke test described in the README.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

FAKE_DIM = 8


def word_token_counter(text: str) -> int:
    """Stand-in for a tokenizer: one token per whitespace-separated word.

    Chunking must be correct with respect to *a* budget; which tokenizer
    supplies the count is not what these tests are checking.
    """
    return len(text.split())


class FakeEmbedder:
    """Deterministic hash-based embedder producing unit vectors."""

    name = "fake-embedder"
    dim = FAKE_DIM

    def encode(self, texts: list[str]) -> np.ndarray:
        """Map each text to a stable unit vector."""
        if not texts:
            return np.zeros((0, FAKE_DIM), dtype=np.float32)
        vectors = np.stack([self._vector(text) for text in texts])
        return np.ascontiguousarray(vectors.astype(np.float32))

    def count_tokens(self, text: str) -> int:
        """Delegate to the injected word counter."""
        return word_token_counter(text)

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        """Derive a unit vector from a digest of the text."""
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = np.frombuffer(digest[:FAKE_DIM], dtype=np.uint8).astype(np.float32)
        centred = raw - raw.mean()
        norm = np.linalg.norm(centred)
        if norm == 0:
            centred = np.ones(FAKE_DIM, dtype=np.float32)
            norm = np.linalg.norm(centred)
        return centred / norm


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    """A model-free embedder."""
    return FakeEmbedder()


@pytest.fixture
def count_tokens():
    """The injected token counter used by the chunking tests."""
    return word_token_counter
