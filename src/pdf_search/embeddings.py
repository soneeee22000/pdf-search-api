"""The embedding seam and its sentence-transformers implementation.

The `Embedder` protocol exists so the test suite can run without downloading a
470 MB model, and so an alternative encoder can be swapped in without touching
the pipeline. The token counter is exposed alongside it because chunk sizing is
a property of the model, not of the chunker -- and so is the window the chunks
have to fit, which is why `budget_for` lives here too.

Queries and passages are encoded through *separate* methods. Retrieval-trained
models are asymmetric: E5 wants "query: " on one side and "passage: " on the
other, Solon wants a prefix on the query alone. Omitting them is silent -- no
error, just quietly worse retrieval -- and applying them to one side only is
worse than omitting them from both. A single `encode` cannot express that
distinction, so it does not exist here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Every sequence carries two special tokens, so a model's advertised window is
# two tokens larger than the content it can actually hold.
SPECIAL_TOKEN_ALLOWANCE = 2

# Deliberate slack below the true capacity. Token counts are measured on the
# assembled chunk, but a tokenizer may segment a rejoined string fractionally
# differently from its parts; this is the margin for that.
BUDGET_HEADROOM_TOKENS = 16

_EMBED_BATCH_SIZE = 32


@dataclass(frozen=True)
class PrefixScheme:
    """The strings a model expects in front of a query and of a passage.

    The default is the conservative one -- no prefixes -- which is correct for
    the incumbent and for any model whose requirements are undocumented.
    """

    query: str = ""
    document: str = ""

    def for_documents(self, texts: list[str]) -> list[str]:
        """Prefix passages, leaving them untouched when the scheme is empty."""
        if not self.document:
            return list(texts)
        return [f"{self.document}{text}" for text in texts]

    def for_query(self, text: str) -> str:
        """Prefix a query, leaving it untouched when the scheme is empty."""
        return f"{self.query}{text}" if self.query else text


# Keyed by exact model id, taken from the model cards. Where a card documents no
# prefix the entry is the empty scheme rather than a guess.
_PREFIX_SCHEMES: dict[str, PrefixScheme] = {
    DEFAULT_MODEL_NAME: PrefixScheme(),
    "intfloat/multilingual-e5-small": PrefixScheme(query="query: ", document="passage: "),
    "intfloat/multilingual-e5-base": PrefixScheme(query="query: ", document="passage: "),
    "intfloat/multilingual-e5-large": PrefixScheme(query="query: ", document="passage: "),
    # Solon documents a space before the colon, and no passage prefix.
    "OrdalieTech/Solon-embeddings-base-0.1": PrefixScheme(query="query : "),
    "OrdalieTech/Solon-embeddings-large-0.1": PrefixScheme(query="query : "),
    "ibm-granite/granite-embedding-97m-multilingual-r2": PrefixScheme(),
    "BAAI/bge-m3": PrefixScheme(),
    "Alibaba-NLP/gte-multilingual-base": PrefixScheme(),
}


def prefixes_for(model_name: str) -> PrefixScheme:
    """Return the documented prefix scheme for a model, or the empty one.

    An unlisted model is encoded without prefixes and the manifest records that,
    so the index and the queries against it stay consistent with each other. The
    cost of guessing wrong here is silent, so nothing is guessed.
    """
    scheme = _PREFIX_SCHEMES.get(model_name)
    if scheme is None:
        logger.warning(
            "no documented prefix scheme for %s; encoding without prefixes. "
            "If this model expects them, retrieval quality degrades silently.",
            model_name,
        )
        return PrefixScheme()
    return scheme


def budget_for(max_seq_length: int) -> int:
    """Content tokens a model can hold, less special tokens and headroom.

    Returns 110 for the incumbent's 128-token window, which is the constant the
    chunker used before this became a function of the model.
    """
    budget = max_seq_length - SPECIAL_TOKEN_ALLOWANCE - BUDGET_HEADROOM_TOKENS
    if budget < 1:
        raise ValueError(
            f"a {max_seq_length}-token window leaves no room for content after "
            f"{SPECIAL_TOKEN_ALLOWANCE} special tokens and "
            f"{BUDGET_HEADROOM_TOKENS} tokens of headroom"
        )
    return budget


@runtime_checkable
class Embedder(Protocol):
    """Turns text into L2-normalised float32 vectors."""

    @property
    def name(self) -> str:
        """Model identifier recorded in the manifest."""

    @property
    def dim(self) -> int:
        """Embedding dimension."""

    @property
    def max_seq_length(self) -> int:
        """Tokens the model accepts, including special tokens."""

    @property
    def prefixes(self) -> PrefixScheme:
        """The query and passage prefixes this model was trained with."""

    def encode_documents(self, texts: list[str]) -> NDArray[np.float32]:
        """Encode passages into a (len(texts), dim) array of unit vectors."""

    def encode_query(self, text: str) -> NDArray[np.float32]:
        """Encode one query into a (1, dim) array of unit vectors."""

    def count_tokens(self, text: str) -> int:
        """Token count under this model's tokenizer."""


class SentenceTransformerEmbedder:
    """Wraps a local sentence-transformers model, loaded once."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        """Load the model. Imported lazily so the package imports without torch."""
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        # Renamed in sentence-transformers 5.x; support both so the console
        # stays clean across versions.
        dimension_of = getattr(
            self._model, "get_embedding_dimension", None
        ) or self._model.get_sentence_embedding_dimension
        self._dim = int(dimension_of())
        self._prefixes = prefixes_for(model_name)

    @property
    def name(self) -> str:
        """Model identifier."""
        return self._model_name

    @property
    def dim(self) -> int:
        """Embedding dimension, read from the loaded model."""
        return self._dim

    @property
    def max_seq_length(self) -> int:
        """The window sentence-transformers will actually honour.

        Read from the loaded model rather than hard-coded, because this is the
        number that silently truncates and it differs per model. A model's
        tokenizer config often advertises a larger figure; sentence-transformers
        uses this one.
        """
        return int(self._model.max_seq_length)

    @property
    def prefixes(self) -> PrefixScheme:
        """The documented prefix scheme for this model."""
        return self._prefixes

    def encode_documents(self, texts: list[str]) -> NDArray[np.float32]:
        """Encode passages, applying the passage-side prefix."""
        return self._encode(self._prefixes.for_documents(texts))

    def encode_query(self, text: str) -> NDArray[np.float32]:
        """Encode one query, applying the query-side prefix."""
        return self._encode([self._prefixes.for_query(text)])

    def _encode(self, texts: list[str]) -> NDArray[np.float32]:
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
        """Count tokens with the model's own tokenizer, excluding special tokens.

        The chunker measures a whole page before deciding how to split it, so the
        text passed here is routinely longer than the model can encode. That is
        the question being asked, not a mistake: nothing is embedded at this
        point. `verbose=False` suppresses the tokenizer's "longer than the
        maximum sequence length" advisory, which would otherwise warn about
        indexing errors that cannot happen -- every chunk this count produces is
        checked against the budget before it reaches the encoder.
        """
        return len(
            self._model.tokenizer.encode(text, add_special_tokens=False, verbose=False)
        )
