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
from typing import Callable, Iterable

from pdf_search.schemas import ChunkRecord, PageRecord

TokenCounter = Callable[[str], int]

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
    if page.status != "extracted" or not page.text.strip():
        return []

    pieces = _split_to_budget(page.text, count_tokens, budget)
    merged = _merge_with_overlap(pieces, count_tokens, budget, overlap)

    records: list[ChunkRecord] = []
    for offset, text in enumerate(merged):
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
            current = _carry_overlap(current, count_tokens, overlap)
            current = f"{current} {span}".strip() if current else span
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if len(c) >= MIN_CHUNK_CHARS or len(chunks) == 1]


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
