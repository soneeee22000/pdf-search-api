"""Data contracts shared by the ingestion pipeline and the search API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

PageStatus = Literal["extracted", "no_text", "error"]

MIN_TOP_K = 1
MAX_TOP_K = 20
DEFAULT_TOP_K = 5


class PageRecord(BaseModel):
    """One page of one PDF, emitted whether or not text was recovered.

    A page that yields nothing is recorded rather than dropped, so the ingestion
    summary describes the whole corpus instead of only the parts that worked.
    """

    document_name: str
    page_number: int = Field(ge=1, description="1-based, matching the reader's page numbering")
    status: PageStatus
    char_count: int = Field(ge=0)
    text: str = ""


class ChunkRecord(BaseModel):
    """An embeddable passage, always contained within a single page."""

    document_name: str
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0, description="Position within the corpus, stable across a rebuild")
    text: str
    token_count: int = Field(ge=0)


class Manifest(BaseModel):
    """Describes the index so the API can refuse to serve one it cannot honour."""

    model_name: str
    embedding_dim: int = Field(gt=0)
    metric: Literal["ip"] = "ip"
    normalized: bool = True
    chunk_token_budget: int = Field(gt=0)
    n_documents: int = Field(ge=0)
    n_chunks: int = Field(ge=0)
    n_pages: int = Field(ge=0)
    n_pages_no_text: int = Field(ge=0)
    created_at: str
    source_documents: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    """A `POST /search` body."""

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=MIN_TOP_K, le=MAX_TOP_K)

    @field_validator("query")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        """Reject a query that is only whitespace."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must contain non-whitespace characters")
        return stripped


class SearchResult(BaseModel):
    """One hit. Field names are fixed by the exercise specification."""

    document_name: str
    page_number: int
    chunk_index: int
    score: float
    text: str


class SearchResponse(BaseModel):
    """The `POST /search` envelope."""

    query: str
    results: list[SearchResult]


class HealthResponse(BaseModel):
    """The `GET /health` payload."""

    status: Literal["ok", "unavailable"]
    n_chunks: int
    model_name: str | None = None
    detail: str | None = None
