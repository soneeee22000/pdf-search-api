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


def test_pages_recovered_by_ocr_are_chunked_like_any_other(count_tokens) -> None:
    """A recognised page must reach the index, not just the summary.

    The chunker originally tested `status != "extracted"`, so every page OCR
    recovered was read correctly and then dropped on the floor. Nothing failed:
    ingestion reported the pages as recovered, the text was right, and the
    document stayed invisible to search. Only an end-to-end retrieval check
    caught it, which is why the status set is asserted here directly.
    """
    page = PageRecord(
        document_name="scan.pdf",
        page_number=1,
        status="ocr",
        char_count=40,
        text="Construction d'un abri de jardin sur la parcelle BE420.",
    )

    chunks = chunk_page(page, count_tokens)

    assert chunks, "a recognised page produced no chunks"
    assert chunks[0].page_number == 1
    assert "BE420" in chunks[0].text


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


def _words(text: str) -> list[str]:
    """Whitespace-delimited words with the separators the splitter consumes removed.

    Byte-exact conservation is not achievable by construction: `_split_to_budget`
    splits on its separators, which consumes them, and the merge rejoins on a
    single space. What must be conserved is the content between separators.
    """
    return [word.strip(".") for word in text.split() if word.strip(".")]


def test_a_short_trailing_span_is_kept_rather_than_dropped(count_tokens) -> None:
    """A bare reference number is short, not meaningless.

    Regression: any chunk under MIN_CHUNK_CHARS was discarded whenever a page
    produced two or more, so a lone deliberation number vanished from the index
    while still sitting in the PDF -- a provenance bug for this corpus.
    """
    body = " ".join(f"mot{i}" for i in range(50))
    reference = "N DEL05-06-2026-02"

    chunks = chunk_page(_page(f"{body}\n\n{reference}"), count_tokens, budget=50, overlap=0)

    assert "DEL05-06-2026-02" in " ".join(chunk.text for chunk in chunks)


def test_chunking_conserves_every_word_of_the_page(count_tokens) -> None:
    """No content is dropped, only redistributed."""
    body = " ".join(f"mot{i}" for i in range(50))
    reference = "N DEL05-06-2026-02"
    page_text = f"{body}\n\n{reference}"

    chunks = chunk_page(_page(page_text), count_tokens, budget=50, overlap=0)

    emitted = [word for chunk in chunks for word in _words(chunk.text)]
    assert emitted == _words(page_text)


def test_no_word_is_lost_when_overlap_duplicates_context(count_tokens) -> None:
    """With overlap on, chunks repeat words but must still cover all of them."""
    tail = " ".join(f"fin{i}" for i in range(10))
    para_a = " ".join(f"mot{i}" for i in range(95)) + ". " + tail
    para_b = " ".join(f"autre{i}" for i in range(105))
    page_text = f"{para_a}\n\n{para_b}"

    chunks = chunk_page(_page(page_text), count_tokens, budget=110, overlap=20)

    emitted = {word for chunk in chunks for word in _words(chunk.text)}
    assert set(_words(page_text)) <= emitted
