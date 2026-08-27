"""API contract and failure modes, exercised without loading a real model."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pdf_search import storage
from pdf_search.api import SearchService, app
from pdf_search.schemas import ChunkRecord, Manifest

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
def client(tmp_path: Path, fake_embedder):
    """A TestClient whose service is injected, bypassing model loading."""
    service = _service(tmp_path, fake_embedder)
    with TestClient(app) as test_client:
        # Set after startup: the lifespan resets state and tries a real load.
        test_client.app.state.service = service
        test_client.app.state.startup_error = None
        yield test_client
    app.state.service = None
    app.state.startup_error = None


@pytest.fixture
def unavailable_client():
    """A TestClient with no index loaded."""
    with TestClient(app) as test_client:
        test_client.app.state.service = None
        test_client.app.state.startup_error = "missing index.faiss. Run python -m pdf_search.ingest"
        yield test_client
    app.state.service = None


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
