"""The embedding seam and its sentence-transformers implementation.

The `Embedder` protocol exists so the test suite can run without downloading a
470 MB model, and so an alternative encoder can be swapped in without touching
the pipeline. The token counter is exposed alongside it because chunk sizing is
a property of the model, not of the chunker.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Declared in the model's sentence_bert_config.json. Its tokenizer_config.json
# says 512; sentence-transformers uses this one, and truncates past it silently.
MODEL_MAX_SEQ_LENGTH = 128

_EMBED_BATCH_SIZE = 32


@runtime_checkable
class Embedder(Protocol):
    """Turns text into L2-normalised float32 vectors."""

    @property
    def name(self) -> str:
        """Model identifier recorded in the manifest."""

    @property
    def dim(self) -> int:
        """Embedding dimension."""

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into a (len(texts), dim) float32 array of unit vectors."""

    def count_tokens(self, text: str) -> int:
        """Token count under this model's tokenizer."""


class SentenceTransformerEmbedder:
    """Wraps a local sentence-transformers model, loaded once."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        """Load the model. Imported lazily so the package imports without torch."""
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def name(self) -> str:
        """Model identifier."""
        return self._model_name

    @property
    def dim(self) -> int:
        """Embedding dimension, read from the loaded model."""
        return self._dim

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode with normalisation on, so inner product is cosine similarity.

        `normalize_embeddings` defaults to False; leaving it off silently turns
        retrieval into length-biased maximum-inner-product search.
        """
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=_EMBED_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vectors.astype(np.float32))

    def count_tokens(self, text: str) -> int:
        """Count tokens with the model's own tokenizer, excluding special tokens."""
        return len(self._model.tokenizer.encode(text, add_special_tokens=False))
