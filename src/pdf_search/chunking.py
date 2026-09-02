"""Page-local, token-budgeted chunking.

The budget is not a tuning knob. `paraphrase-multilingual-MiniLM-L12-v2` sets
`max_seq_length` to 128 tokens in its `sentence_bert_config.json`, and
sentence-transformers truncates anything longer *silently*. A chunk sized by
the usual 1000-character folk default would therefore be embedded from its
first ~40% while the API still returned the whole text, so the vector would
describe something other than the passage shown to the caller.

Chunks never span a page boundary, so `page_number` is exact rather than
reconstructed from character offsets.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from pdf_search.schemas import ChunkRecord, PageRecord

# Statuses that carry text worth indexing. A page recovered by OCR is text like
# any other here; the distinction matters to the ingestion summary, not to the
# chunker. Testing against 'extracted' alone silently discarded every recognised
# page -- the text was read correctly and then thrown away, which the fidelity
# metric could not see and only the retrieval metric caught.
CHUNKABLE_STATUSES = ("extracted", "ocr")

TokenCounter = Callable[[str], int]

# The default, and what the incumbent model earns. Ingestion does not use it:
# it derives the budget from the loaded model via `embeddings.budget_for`, so a
# model with a wider window produces larger chunks without a code change. The
# two are pinned to each other by a test.
CHUNK_TOKEN_BUDGET = 110
CHUNK_TOKEN_OVERLAP = 20
MIN_CHUNK_CHARS = 40

# Tried in order: paragraph, line, sentence, then whitespace. The first
# separator that actually splits an oversized span is used.
_SEPARATORS = ("\n\n", "\n", ". ", " ")

_SENTENCE_TAIL = re.compile(r"(?<=[.!?])\s+")


def chunk_page(
    page: PageRecord,
    count_tokens: TokenCounter,
    start_index: int = 0,
    budget: int = CHUNK_TOKEN_BUDGET,
    overlap: int = CHUNK_TOKEN_OVERLAP,
) -> list[ChunkRecord]:
    """Split one page into chunks that each fit inside the token budget.

    `count_tokens` is injected so the unit tests can run without downloading a
    model; ingestion passes the real tokenizer.
    """
    if page.status not in CHUNKABLE_STATUSES or not page.text.strip():
        return []

    pieces = _split_to_budget(page.text, count_tokens, budget)
    merged = _merge_with_overlap(pieces, count_tokens, budget, overlap)
    sized = _enforce_budget(merged, count_tokens, budget)

    records: list[ChunkRecord] = []
    for offset, text in enumerate(sized):
        records.append(
            ChunkRecord(
                document_name=page.document_name,
                page_number=page.page_number,
                chunk_index=start_index + offset,
                text=text,
                token_count=count_tokens(text),
            )
        )
    return records


def _split_to_budget(text: str, count_tokens: TokenCounter, budget: int) -> list[str]:
    """Break text into spans that each fit the budget, preferring natural breaks."""
    if count_tokens(text) <= budget:
        return [text]

    for separator in _SEPARATORS:
        parts = [p for p in text.split(separator) if p.strip()]
        if len(parts) < 2:
            continue
        spans: list[str] = []
        for part in parts:
            if count_tokens(part) <= budget:
                spans.append(part.strip())
            else:
                spans.extend(_split_to_budget(part, count_tokens, budget))
        return spans

    return _hard_split(text, count_tokens, budget)


def _hard_split(text: str, count_tokens: TokenCounter, budget: int) -> list[str]:
    """Last resort for a span with no usable separator: cut it by characters.

    Reached only by pathological input such as an unbroken table row, and the
    ratio is recomputed from the actual text rather than assumed.
    """
    tokens = max(count_tokens(text), 1)
    chars_per_token = max(len(text) / tokens, 1.0)
    window = max(int(budget * chars_per_token), MIN_CHUNK_CHARS)
    return [text[i : i + window].strip() for i in range(0, len(text), window) if text[i : i + window].strip()]


def _merge_with_overlap(
    spans: Iterable[str], count_tokens: TokenCounter, budget: int, overlap: int
) -> list[str]:
    """Recombine small spans up to the budget, carrying a sentence of context."""
    chunks: list[str] = []
    current = ""

    for span in spans:
        candidate = f"{current} {span}".strip() if current else span
        if current and count_tokens(candidate) > budget:
            chunks.append(current)
            carried = _carry_overlap(current, count_tokens, overlap)
            current = f"{carried} {span}".strip() if carried else span
            if count_tokens(current) > budget:
                # Context is worth carrying, but not at the cost of the budget
                # it was carried into: the model would truncate it away anyway.
                current = span
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())

    return _absorb_runts(chunks, count_tokens, budget)


def _absorb_runts(chunks: list[str], count_tokens: TokenCounter, budget: int) -> list[str]:
    """Fold a too-short chunk into its predecessor, or keep it standalone.

    These used to be discarded outright. A bare reference number or a one-cell
    table row is short, not meaningless, and a passage the corpus contains but
    the index does not is a provenance bug -- which is the one thing this
    system is supposed to be trusted on.
    """
    kept: list[str] = []
    for chunk in chunks:
        if kept and len(chunk) < MIN_CHUNK_CHARS:
            merged = f"{kept[-1]} {chunk}".strip()
            if count_tokens(merged) <= budget:
                kept[-1] = merged
                continue
        kept.append(chunk)
    return kept


def _enforce_budget(texts: Iterable[str], count_tokens: TokenCounter, budget: int) -> list[str]:
    """Post-condition: nothing leaves this module over budget.

    Both `_hard_split` and the overlap merge work from estimates, so the budget
    is re-checked here against the real counter rather than trusted. A single
    word longer than the budget is emitted alone -- without the tokenizer
    itself there is no smaller unit to cut on.
    """
    sized: list[str] = []
    for text in texts:
        if count_tokens(text) <= budget:
            sized.append(text)
            continue
        current = ""
        for word in text.split():
            candidate = f"{current} {word}".strip() if current else word
            if current and count_tokens(candidate) > budget:
                sized.append(current)
                current = word
            else:
                current = candidate
        if current:
            sized.append(current)
    return sized


def _carry_overlap(text: str, count_tokens: TokenCounter, overlap: int) -> str:
    """Return the trailing sentences of `text` that fit within the overlap budget."""
    if overlap <= 0:
        return ""
    sentences = _SENTENCE_TAIL.split(text)
    carried = ""
    for sentence in reversed(sentences):
        candidate = f"{sentence} {carried}".strip() if carried else sentence
        if count_tokens(candidate) > overlap:
            break
        carried = candidate
    return carried
