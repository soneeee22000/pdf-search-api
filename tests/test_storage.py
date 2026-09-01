"""Index persistence, score orientation, and the consistency guards."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdf_search import storage
from pdf_search.schemas import ChunkRecord, Manifest
from pdf_search.storage import IndexUnavailableError

TEXTS = [
    "Le conseil municipal approuve la convention de mecenat financier.",
    "Declaration prealable accordee pour la place des Hauts Taillis.",
    "Ordre du jour de la seance du 18 juin 2026.",
]


def _chunks() -> list[ChunkRecord]:
    """Three chunks spread over two documents."""
    return [
        ChunkRecord(
            document_name="doc-a.pdf" if i < 2 else "doc-b.pdf",
            page_number=i + 1,
            chunk_index=i,
            text=text,
            token_count=len(text.split()),
        )
        for i, text in enumerate(TEXTS)
    ]


def _manifest(dim: int, n_chunks: int) -> Manifest:
    """A manifest matching the given index shape."""
    return Manifest(
        model_name="fake-embedder",
        embedding_dim=dim,
        chunk_token_budget=110,
        n_documents=2,
        n_chunks=n_chunks,
        n_pages=3,
        n_pages_no_text=0,
        created_at="2026-08-27T18:00:00+00:00",
        source_documents=["doc-a.pdf", "doc-b.pdf"],
    )


def _save(tmp_path: Path, fake_embedder) -> Path:
    """Build and persist a snapshot; return its directory."""
    chunks = _chunks()
    vectors = fake_embedder.encode_documents([c.text for c in chunks])
    index = storage.build_index(vectors, fake_embedder.dim)
    out = tmp_path / "storage"
    storage.save(out, index, chunks, _manifest(fake_embedder.dim, len(chunks)))
    return out


def test_round_trip_preserves_metadata(tmp_path: Path, fake_embedder) -> None:
    """Save then load returns the same chunks in the same order."""
    loaded = storage.load(_save(tmp_path, fake_embedder))

    assert loaded.index.ntotal == 3
    assert [c.text for c in loaded.chunks] == TEXTS
    assert loaded.manifest.model_name == "fake-embedder"


def test_exact_match_scores_highest_and_results_are_descending(
    tmp_path: Path, fake_embedder
) -> None:
    """Higher score means more similar, matching the documented contract."""
    loaded = storage.load(_save(tmp_path, fake_embedder))
    query_vector = fake_embedder.encode_query(TEXTS[1])

    results = loaded.search(query_vector, top_k=3)

    assert results[0].text == TEXTS[1]
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_scores_stay_inside_the_cosine_range(tmp_path: Path, fake_embedder) -> None:
    """Normalised vectors plus inner product give a cosine in [-1, 1]."""
    loaded = storage.load(_save(tmp_path, fake_embedder))
    results = loaded.search(fake_embedder.encode_query("une requete quelconque"), top_k=3)

    assert all(-1.0 <= r.score <= 1.0 for r in results)


def test_top_k_larger_than_the_corpus_returns_everything_without_crashing(
    tmp_path: Path, fake_embedder
) -> None:
    """FAISS pads short result rows with -1; those slots must be skipped."""
    loaded = storage.load(_save(tmp_path, fake_embedder))

    results = loaded.search(fake_embedder.encode_query("convention"), top_k=20)

    assert len(results) == 3


def test_missing_snapshot_raises_an_actionable_error(tmp_path: Path) -> None:
    """The message must name the command that fixes it."""
    with pytest.raises(IndexUnavailableError, match="pdf_search.ingest"):
        storage.load(tmp_path / "absent")


def test_metadata_and_index_disagreement_is_refused(tmp_path: Path, fake_embedder) -> None:
    """A truncated sidecar must fail loudly, not silently misattribute results."""
    directory = _save(tmp_path, fake_embedder)
    metadata = directory / storage.METADATA_FILENAME
    kept = metadata.read_text(encoding="utf-8").splitlines()[:2]
    metadata.write_text("\n".join(kept) + "\n", encoding="utf-8")

    with pytest.raises(IndexUnavailableError, match="inconsistent"):
        storage.load(directory)


def test_dimension_mismatch_is_refused(tmp_path: Path, fake_embedder) -> None:
    """An index built by a different model must not be served."""
    directory = _save(tmp_path, fake_embedder)
    manifest_path = directory / storage.MANIFEST_FILENAME
    manifest = Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(
        manifest.model_copy(update={"embedding_dim": 384}).model_dump_json(), encoding="utf-8"
    )

    with pytest.raises(IndexUnavailableError, match="dimension"):
        storage.load(directory)


def test_save_refuses_a_vector_count_that_disagrees_with_the_chunks(
    tmp_path: Path, fake_embedder
) -> None:
    """The alignment invariant is enforced on write as well as on read."""
    chunks = _chunks()
    vectors = fake_embedder.encode_documents([c.text for c in chunks[:2]])
    index = storage.build_index(vectors, fake_embedder.dim)

    with pytest.raises(ValueError, match="chunks"):
        storage.save(tmp_path / "storage", index, chunks, _manifest(fake_embedder.dim, 3))


def test_save_works_when_the_output_directory_parent_is_not_writable(
    tmp_path: Path, fake_embedder, monkeypatch
) -> None:
    """Regression: staging must live inside output_dir, not beside it.

    In Docker the output directory is a mounted volume whose parent is the
    container root. A sibling staging directory fails there with EACCES, which
    is exactly what the first real container run hit.
    """
    chunks = _chunks()
    vectors = fake_embedder.encode_documents([c.text for c in chunks])
    index = storage.build_index(vectors, fake_embedder.dim)
    out = tmp_path / "mounted"
    out.mkdir()

    real_mkdir = Path.mkdir

    def deny_outside(self, *args, **kwargs):
        if self.parent == tmp_path and self != out:
            raise PermissionError(13, "Permission denied", str(self))
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", deny_outside)
    storage.save(out, index, chunks, _manifest(fake_embedder.dim, len(chunks)))
    monkeypatch.undo()

    assert storage.load(out).index.ntotal == 3


def test_staging_directory_is_cleaned_up(tmp_path: Path, fake_embedder) -> None:
    """A leftover .staging directory would confuse the next run."""
    directory = _save(tmp_path, fake_embedder)

    assert not (directory / ".staging").exists()
    assert sorted(p.name for p in directory.iterdir()) == [
        storage.INDEX_FILENAME,
        storage.MANIFEST_FILENAME,
        storage.METADATA_FILENAME,
    ]


def test_reordered_metadata_is_refused(tmp_path: Path, fake_embedder) -> None:
    """Line order in the sidecar *is* index row order; a reshuffle must not pass.

    Regression: count and dimension both still matched, so a reordered sidecar
    loaded cleanly and every result carried confident, wrong provenance -- the
    worst possible failure for a tool whose product is traceability.
    """
    out = _save(tmp_path, fake_embedder)
    metadata = out / storage.METADATA_FILENAME
    lines = metadata.read_text(encoding="utf-8").splitlines()
    metadata.write_text("\n".join(reversed(lines)) + "\n", encoding="utf-8")

    with pytest.raises(IndexUnavailableError):
        storage.load(out)


def test_an_index_swapped_for_another_of_the_same_shape_is_refused(
    tmp_path: Path, fake_embedder
) -> None:
    """Matching row count and dimension are not evidence of matching content."""
    out = _save(tmp_path, fake_embedder)
    other = storage.build_index(
        fake_embedder.encode_documents(["texte sans rapport"] * len(TEXTS)), fake_embedder.dim
    )
    import faiss

    faiss.write_index(other, str(out / storage.INDEX_FILENAME))

    with pytest.raises(IndexUnavailableError):
        storage.load(out)


def test_a_snapshot_without_digests_is_refused(tmp_path: Path, fake_embedder) -> None:
    """An index written before the digests existed cannot be verified, so it is not served."""
    out = _save(tmp_path, fake_embedder)
    manifest_path = out / storage.MANIFEST_FILENAME
    stale = _manifest(fake_embedder.dim, len(TEXTS))
    manifest_path.write_text(stale.model_dump_json(indent=2), encoding="utf-8")

    with pytest.raises(IndexUnavailableError, match="digest"):
        storage.load(out)
