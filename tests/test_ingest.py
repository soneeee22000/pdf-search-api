"""Ingestion orchestration: discovery guards and cross-page chunk numbering."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdf_search.ingest import build_argument_parser, collect_chunks, run_ingestion
from pdf_search.pdf_text import discover_pdfs
from pdf_search.schemas import PageRecord


def test_discovery_rejects_a_missing_directory(tmp_path: Path) -> None:
    """A wrong path must fail immediately with a clear error."""
    with pytest.raises(NotADirectoryError):
        discover_pdfs(tmp_path / "absent")


def test_discovery_rejects_an_empty_directory(tmp_path: Path) -> None:
    """An empty corpus must not produce an empty index."""
    with pytest.raises(FileNotFoundError, match="no PDF files"):
        discover_pdfs(tmp_path)


def test_discovery_is_sorted_and_ignores_non_pdfs(tmp_path: Path) -> None:
    """Deterministic order means a rebuild produces identical chunk indices."""
    for name in ("b.pdf", "a.pdf", "notes.txt", "C.PDF"):
        (tmp_path / name).write_bytes(b"%PDF-1.4\n")

    # Case-folded ordering, identical on Windows and Linux.
    assert [p.name for p in discover_pdfs(tmp_path)] == ["a.pdf", "b.pdf", "C.PDF"]


def test_chunk_indices_are_contiguous_across_pages(fake_embedder) -> None:
    """Chunk numbering is corpus-wide, not per page."""
    pages = [
        PageRecord(
            document_name="doc.pdf",
            page_number=n,
            status="extracted",
            char_count=200,
            text=" ".join(f"mot{i}" for i in range(150)),
        )
        for n in (1, 2, 3)
    ]

    chunks = collect_chunks(pages, fake_embedder)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert {c.page_number for c in chunks} == {1, 2, 3}


def test_ingestion_fails_loudly_when_nothing_could_be_extracted(
    tmp_path: Path, fake_embedder, monkeypatch
) -> None:
    """A corpus of scans must not silently produce an empty index."""
    (tmp_path / "scan.pdf").write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(
        "pdf_search.ingest.extract_pages",
        lambda path: [
            PageRecord(document_name=path.name, page_number=1, status="no_text", char_count=0)
        ],
    )

    with pytest.raises(RuntimeError, match="OCR"):
        run_ingestion(tmp_path, tmp_path / "storage", fake_embedder)


def test_input_dir_is_a_required_argument() -> None:
    """The exercise mandates the PDF folder path as a command-line argument."""
    with pytest.raises(SystemExit):
        build_argument_parser().parse_args([])
