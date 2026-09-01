"""API contract and failure modes, exercised without loading a real model."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pdf_search import api, storage
from pdf_search.api import SearchService, app
from pdf_search.schemas import ChunkRecord, Manifest
from pdf_search.storage import IndexUnavailableError

TEXTS = [
    "Le conseil municipal approuve la convention de mecenat financier.",
    "Declaration prealable accordee pour la place des Hauts Taillis.",
]


def _service(tmp_path: Path, fake_embedder) -> SearchService:
    """Build a service backed by a real on-disk snapshot and a fake embedder."""
    chunks = [
        ChunkRecord(
            document_name="doc-a.pdf",
            page_number=i + 1,
            chunk_index=i,
            text=text,
            token_count=len(text.split()),
        )
        for i, text in enumerate(TEXTS)
    ]
    vectors = fake_embedder.encode([c.text for c in chunks])
    index = storage.build_index(vectors, fake_embedder.dim)
    manifest = Manifest(
        model_name=fake_embedder.name,
        embedding_dim=fake_embedder.dim,
        chunk_token_budget=110,
        n_documents=1,
        n_chunks=len(chunks),
        n_pages=2,
        n_pages_no_text=0,
        created_at="2026-08-27T18:00:00+00:00",
        source_documents=["doc-a.pdf"],
    )
    out = tmp_path / "storage"
    storage.save(out, index, chunks, manifest)
    return SearchService(storage.load(out), fake_embedder)


@pytest.fixture
def client(tmp_path: Path, fake_embedder, monkeypatch):
    """A TestClient whose startup builds the fake service instead of a real one.

    Patched *before* the context manager, because that is what runs the
    lifespan: injecting afterwards left the real model and the real storage
    directory to be loaded first, which made the suite slow and dependent on
    whatever happened to be on disk.
    """
    service = _service(tmp_path, fake_embedder)
    monkeypatch.setattr(api, "_build_service", lambda: service)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def unavailable_client(monkeypatch):
    """A TestClient whose startup fails to find an index.

    This drives the real failure path in the lifespan rather than simulating
    its result, so the error handling itself is covered.
    """

    def _fail() -> SearchService:
        raise IndexUnavailableError("missing index.faiss. Run python -m pdf_search.ingest")

    monkeypatch.setattr(api, "_build_service", _fail)
    with TestClient(app) as test_client:
        yield test_client


def test_search_returns_the_five_specified_fields(client) -> None:
    """The response schema is fixed by the exercise specification."""
    response = client.post("/search", json={"query": "convention de mecenat", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "convention de mecenat"
    assert len(body["results"]) == 2
    assert set(body["results"][0]) == {
        "document_name",
        "page_number",
        "chunk_index",
        "score",
        "text",
    }


def test_results_are_sorted_by_descending_score(client) -> None:
    """Higher is better, matching the contract's example."""
    body = client.post("/search", json={"query": TEXTS[0], "top_k": 2}).json()
    scores = [r["score"] for r in body["results"]]

    assert scores == sorted(scores, reverse=True)


def test_top_k_defaults_to_five(client) -> None:
    """top_k is optional."""
    body = client.post("/search", json={"query": "conseil"}).json()

    assert len(body["results"]) == len(TEXTS)


def test_blank_query_is_rejected(client) -> None:
    """A whitespace-only query is a client error, not an empty result set."""
    assert client.post("/search", json={"query": "   "}).status_code == 422


def test_missing_query_is_rejected(client) -> None:
    """The query field is required."""
    assert client.post("/search", json={"top_k": 3}).status_code == 422


@pytest.mark.parametrize("top_k", [0, -1, 21, 100])
def test_top_k_outside_the_supported_range_is_rejected(client, top_k: int) -> None:
    """Bounding top_k also removes the FAISS empty-slot case by construction."""
    assert client.post("/search", json={"query": "conseil", "top_k": top_k}).status_code == 422


def test_health_reports_a_loaded_index(client) -> None:
    """Health is the one-command Docker verification step."""
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["n_chunks"] == len(TEXTS)


def test_search_without_an_index_returns_an_actionable_503(unavailable_client) -> None:
    """The message must name the ingestion command."""
    response = unavailable_client.post("/search", json={"query": "conseil"})

    assert response.status_code == 503
    assert "ingest" in response.json()["detail"]


def test_health_without_an_index_reports_unavailable(unavailable_client) -> None:
    """Health degrades rather than erroring, so the container stays diagnosable."""
    body = unavailable_client.get("/health").json()

    assert body["status"] == "unavailable"
    assert body["n_chunks"] == 0


def test_a_model_disagreeing_with_the_manifest_is_refused(
    tmp_path: Path, fake_embedder, monkeypatch
) -> None:
    """Two models can share a dimension while embedding into different spaces.

    Regression: the override was honoured whenever the dimension matched, so
    queries were embedded by one model and compared against documents embedded
    by another. Every score was meaningless and nothing reported a problem.
    """
    _service(tmp_path, fake_embedder)
    monkeypatch.setenv(api.STORAGE_DIR_ENV, str(tmp_path / "storage"))
    monkeypatch.setenv(api.MODEL_NAME_ENV, "another-model-of-the-same-width")

    with pytest.raises(IndexUnavailableError, match="different vector space"):
        api._build_service()
