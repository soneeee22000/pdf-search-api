"""Chunking behaviour: the token budget, page provenance, and index stability."""

from __future__ import annotations

from pdf_search.chunking import chunk_page
from pdf_search.schemas import PageRecord


def _page(text: str, page_number: int = 1, name: str = "doc.pdf") -> PageRecord:
    """Build an extracted page record for testing."""
    return PageRecord(
        document_name=name,
        page_number=page_number,
        status="extracted",
        char_count=len(text),
        text=text,
    )


def test_no_chunk_exceeds_the_token_budget(count_tokens) -> None:
    """Every chunk must fit the embedding model's window."""
    text = " ".join(f"mot{i}" for i in range(600))
    chunks = chunk_page(_page(text), count_tokens, budget=110, overlap=20)

    assert chunks
    assert all(count_tokens(chunk.text) <= 110 for chunk in chunks)


def test_short_page_yields_a_single_chunk(count_tokens) -> None:
    """Text already inside the budget is not split."""
    chunks = chunk_page(_page("Le conseil municipal approuve la convention de mecenat."), count_tokens)

    assert len(chunks) == 1
    assert chunks[0].text.startswith("Le conseil municipal")


def test_page_number_is_carried_onto_every_chunk(count_tokens) -> None:
    """Provenance is exact because chunks never span a page boundary."""
    text = " ".join(f"mot{i}" for i in range(400))
    chunks = chunk_page(_page(text, page_number=7), count_tokens, budget=50)

    assert chunks
    assert {chunk.page_number for chunk in chunks} == {7}
    assert {chunk.document_name for chunk in chunks} == {"doc.pdf"}


def test_chunk_indices_are_contiguous_from_the_start_index(count_tokens) -> None:
    """Chunk indices continue across pages without gaps."""
    text = " ".join(f"mot{i}" for i in range(300))
    chunks = chunk_page(_page(text), count_tokens, start_index=12, budget=50)

    assert [chunk.chunk_index for chunk in chunks] == list(range(12, 12 + len(chunks)))


def test_chunking_is_deterministic(count_tokens) -> None:
    """A rebuild produces byte-identical chunks."""
    text = " ".join(f"mot{i}" for i in range(300))
    first = chunk_page(_page(text), count_tokens, budget=50)
    second = chunk_page(_page(text), count_tokens, budget=50)

    assert [c.text for c in first] == [c.text for c in second]


def test_pages_without_text_produce_no_chunks(count_tokens) -> None:
    """A page with no text layer contributes nothing but is not an error."""
    page = PageRecord(document_name="scan.pdf", page_number=1, status="no_text", char_count=0)

    assert chunk_page(page, count_tokens) == []


def test_paragraph_boundaries_are_preferred_over_mid_sentence_cuts(count_tokens) -> None:
    """Natural separators are used before falling back to a hard split."""
    paragraph_a = " ".join(f"alpha{i}" for i in range(60))
    paragraph_b = " ".join(f"beta{i}" for i in range(60))
    chunks = chunk_page(_page(f"{paragraph_a}\n\n{paragraph_b}"), count_tokens, budget=70, overlap=0)

    assert len(chunks) >= 2
    assert not any("alpha" in c.text and "beta" in c.text for c in chunks)


def test_carried_overlap_cannot_push_a_chunk_over_the_budget(count_tokens) -> None:
    """The overlap is carried within the budget, never added on top of it.

    Regression: a paragraph split preserves its internal sentence punctuation,
    so the carry is non-empty and used to be prepended to a fresh full-size
    span without re-checking. The result was a chunk of budget + overlap tokens
    that the model would silently truncate -- the exact failure this module
    exists to prevent.
    """
    tail = " ".join(f"fin{i}" for i in range(10))
    para_a = " ".join(f"mot{i}" for i in range(95)) + ". " + tail
    para_b = " ".join(f"autre{i}" for i in range(105))

    chunks = chunk_page(_page(f"{para_a}\n\n{para_b}"), count_tokens, budget=110, overlap=20)

    assert chunks
    assert max(count_tokens(chunk.text) for chunk in chunks) <= 110
