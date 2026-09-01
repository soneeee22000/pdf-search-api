"""The search API.

The model and index are loaded once at startup. A missing or inconsistent index
produces a 503 that names the command to fix it, rather than a traceback or an
empty result set that looks like a legitimate 'no matches'.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import Depends, FastAPI, HTTPException, Request, status

from pdf_search import storage
from pdf_search.embeddings import DEFAULT_MODEL_NAME, Embedder
from pdf_search.schemas import HealthResponse, Manifest, SearchRequest, SearchResponse
from pdf_search.storage import IndexUnavailableError, LoadedIndex

logger = logging.getLogger(__name__)

STORAGE_DIR_ENV = "PDF_SEARCH_STORAGE_DIR"
MODEL_NAME_ENV = "PDF_SEARCH_MODEL"


class SearchService:
    """Holds the loaded index and embedder for the lifetime of the process."""

    def __init__(self, loaded: LoadedIndex, embedder: Embedder) -> None:
        """Bind an index to the embedder that produced it."""
        self._loaded = loaded
        self._embedder = embedder

    @property
    def n_chunks(self) -> int:
        """Number of indexed chunks."""
        return len(self._loaded.chunks)

    @property
    def model_name(self) -> str:
        """The model recorded in the manifest."""
        return self._loaded.manifest.model_name

    def search(self, query: str, top_k: int) -> SearchResponse:
        """Embed the query and return the most similar chunks."""
        query_vector = self._embedder.encode([query])
        results = self._loaded.search(query_vector, top_k)
        return SearchResponse(query=query, results=results)


def _storage_dir() -> Path:
    """Resolve the snapshot directory from the environment."""
    return Path(os.environ.get(STORAGE_DIR_ENV, "storage"))


def _resolve_model_name(manifest: Manifest) -> str:
    """The manifest decides which model may serve its index.

    An override that disagrees is refused rather than honoured. Two different
    models can share an embedding width, so the dimension check below would
    pass while every query vector landed in a different space from the
    documents -- scores would still look plausible and nothing would report a
    problem. Checked before the model is constructed, so the refusal costs
    nothing and does not depend on the wrong model being unavailable.
    """
    recorded = manifest.model_name or DEFAULT_MODEL_NAME
    requested = os.environ.get(MODEL_NAME_ENV)
    if requested and requested != recorded:
        raise IndexUnavailableError(
            f"{MODEL_NAME_ENV} is set to {requested} but this index was built with {recorded}. "
            "Queries would be embedded into a different vector space from the documents. "
            "Re-run ingestion with the model you want, or unset the variable."
        )
    return recorded


def _build_service() -> SearchService:
    """Load the snapshot and the matching model. Raises IndexUnavailableError."""
    from pdf_search.embeddings import SentenceTransformerEmbedder

    loaded = storage.load(_storage_dir())
    model_name = _resolve_model_name(loaded.manifest)
    embedder = SentenceTransformerEmbedder(model_name)

    if embedder.dim != loaded.manifest.embedding_dim:
        raise IndexUnavailableError(
            f"model {model_name} produces {embedder.dim}-d vectors but the index was built "
            f"with {loaded.manifest.embedding_dim}-d vectors. Rebuild the index with this model."
        )
    return SearchService(loaded, embedder)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the index and model once, recording failure instead of crashing.

    A container that exits on a missing index is harder to diagnose than one
    that starts and explains itself on /health.
    """
    app.state.service = None
    app.state.startup_error = None
    try:
        app.state.service = _build_service()
        logger.info("loaded %d chunks", app.state.service.n_chunks)
    except (IndexUnavailableError, OSError) as exc:
        app.state.startup_error = str(exc)
        logger.error("index unavailable: %s", exc)
    yield


def get_service(request: Request) -> SearchService:
    """Return the loaded service or raise an actionable 503."""
    service = getattr(request.app.state, "service", None)
    if service is None:
        detail = getattr(request.app.state, "startup_error", None) or "index not loaded"
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
    return service


app = FastAPI(
    title="PDF Search API",
    description="Similarity search over chunks extracted from a local PDF corpus.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Report whether an index is loaded, and how large it is.

    Degrades rather than erroring, so a container started without an index is
    still diagnosable with one command.
    """
    service = getattr(request.app.state, "service", None)
    if service is None:
        return HealthResponse(
            status="unavailable",
            n_chunks=0,
            detail=getattr(request.app.state, "startup_error", None) or "index not loaded",
        )
    return HealthResponse(status="ok", n_chunks=service.n_chunks, model_name=service.model_name)


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest, service: SearchService = Depends(get_service)) -> SearchResponse:
    """Return the chunks most similar to the query, most similar first.

    Scores are cosine similarities in [-1, 1]; higher is more similar.
    """
    return service.search(request.query, request.top_k)
