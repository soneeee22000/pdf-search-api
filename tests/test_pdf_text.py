"""Extraction against real PDF files.

The fixtures are tiny generated PDFs, committed so the suite needs no network.
`tests/fixtures/README.md` holds the script that produced them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pdf_search.chunking import chunk_page
from pdf_search.pdf_text import extract_pages

FIXTURES = Path(__file__).parent / "fixtures"


def test_pages_are_numbered_from_one_in_reading_order() -> None:
    """A one-off in page_number is the most visible possible provenance bug."""
    pages = extract_pages(FIXTURES / "deliberation_fr.pdf")

    assert [page.page_number for page in pages] == [1, 2]
    assert all(page.document_name == "deliberation_fr.pdf" for page in pages)


def test_text_is_extracted_and_attributed_to_the_right_page() -> None:
    """Content from page 2 must not be attributed to page 1."""
    pages = extract_pages(FIXTURES / "deliberation_fr.pdf")

    assert "REGISTRE DES DELIBERATIONS" in pages[0].text
    assert "ORDRE DU JOUR" in pages[1].text
    assert "ORDRE DU JOUR" not in pages[0].text


def test_extracted_pages_are_marked_extracted() -> None:
    """A page with a text layer is usable."""
    pages = extract_pages(FIXTURES / "deliberation_fr.pdf")

    assert all(page.status == "extracted" for page in pages)
    assert all(page.char_count > 0 for page in pages)


def test_a_page_without_a_text_layer_is_reported_not_dropped() -> None:
    """This is the scanned-document case: recorded, never silently skipped."""
    pages = extract_pages(FIXTURES / "no_text_layer.pdf")

    assert len(pages) == 1
    assert pages[0].status == "no_text"
    assert pages[0].char_count == 0


def test_a_corrupt_file_yields_an_error_record_rather_than_raising(tmp_path: Path) -> None:
    """One unreadable file must not abort ingestion of the whole corpus."""
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4\nthis is not a pdf")

    pages = extract_pages(broken)

    assert len(pages) == 1
    assert pages[0].status == "error"


def test_legal_citations_survive_extraction_and_chunking(count_tokens) -> None:
    """'L. 2121-29' is the shape dense retrieval handles worst; it must at least survive."""
    pages = extract_pages(FIXTURES / "deliberation_fr.pdf")
    chunks = [c for page in pages for c in chunk_page(page, count_tokens)]

    assert any("2121-29" in chunk.text for chunk in chunks)


def test_every_chunk_from_a_real_pdf_respects_the_budget(count_tokens) -> None:
    """The budget assertion holds on real extracted text, not only synthetic input."""
    pages = extract_pages(FIXTURES / "deliberation_fr.pdf")
    chunks = [c for page in pages for c in chunk_page(page, count_tokens, budget=40)]

    assert chunks
    assert all(count_tokens(chunk.text) <= 40 for chunk in chunks)


@pytest.mark.parametrize("name", ["deliberation_fr.pdf", "no_text_layer.pdf"])
def test_extraction_never_raises_on_a_fixture(name: str) -> None:
    """Extraction is total: every page produces a record with some status."""
    pages = extract_pages(FIXTURES / name)

    assert pages
    assert all(page.status in {"extracted", "no_text", "error"} for page in pages)
