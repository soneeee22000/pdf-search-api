"""Test fixtures.

Everything here runs without network access and without downloading a model:
the embedder is faked and the token counter is injected. The real tokenizer is
exercised once, manually, in the ingestion smoke test described in the README.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from pdf_search.embeddings import PrefixScheme

FAKE_DIM = 8
FAKE_MAX_SEQ_LENGTH = 128


def word_token_counter(text: str) -> int:
    """Stand-in for a tokenizer: one token per whitespace-separated word.

    Chunking must be correct with respect to *a* budget; which tokenizer
    supplies the count is not what these tests are checking.
    """
    return len(text.split())


class FakeEmbedder:
    """Deterministic hash-based embedder producing unit vectors.

    The vector is derived from the text *after* prefixing, so a prefixed query
    and the same text as a passage embed differently -- which is what makes the
    asymmetry testable without loading a model.
    """

    name = "fake-embedder"
    dim = FAKE_DIM
    max_seq_length = FAKE_MAX_SEQ_LENGTH

    def __init__(self, prefixes: PrefixScheme | None = None) -> None:
        """Default to the empty scheme, matching the incumbent model."""
        self._prefixes = prefixes or PrefixScheme()

    @property
    def prefixes(self) -> PrefixScheme:
        """The prefix scheme this fake claims to have been trained with."""
        return self._prefixes

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """Map each passage to a stable unit vector."""
        return self._encode(self._prefixes.for_documents(texts))

    def encode_query(self, text: str) -> np.ndarray:
        """Map one query to a stable unit vector."""
        return self._encode([self._prefixes.for_query(text)])

    def count_tokens(self, text: str) -> int:
        """Delegate to the injected word counter."""
        return word_token_counter(text)

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Stack one unit vector per text."""
        if not texts:
            return np.zeros((0, FAKE_DIM), dtype=np.float32)
        vectors = np.stack([self._vector(text) for text in texts])
        return np.ascontiguousarray(vectors.astype(np.float32))

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
    """A model-free embedder with no prefixes, like the incumbent."""
    return FakeEmbedder()


@pytest.fixture
def count_tokens():
    """The injected token counter used by the chunking tests."""
    return word_token_counter
