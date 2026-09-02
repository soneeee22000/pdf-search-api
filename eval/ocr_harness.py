"""Score OCR engines on the pre-registered rule in OCR_DECISION.md.

Two metrics, because they answer different questions. Fidelity asks whether the
engine read the page. Retrievability asks whether the page became findable,
which is the only reason to run OCR at all.

Both are reported strictly and space-insensitively. An engine whose recogniser
has no space token produces text that reads correctly to a human squinting at
it and tokenises into nonsense, so hiding the difference between the two would
hide the defect that actually matters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import tempfile
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval.harness import _hit_rank as embedding_rank  # noqa: E402
from eval.harness import load_queries as load_embedding_queries  # noqa: E402

from pdf_search import storage  # noqa: E402
from pdf_search.embeddings import SentenceTransformerEmbedder  # noqa: E402
from pdf_search.ingest import run_ingestion  # noqa: E402
from pdf_search.ocr import build_engine  # noqa: E402

SEARCH_TOP_K = 10
PRIMARY_K = 5
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class OcrQuery:
    """One string read off the scan, and a question that avoids its wording."""

    document_name: str
    gold_page: int
    gold_string: str
    tier_b_query: str


def load_queries(path: Path) -> list[OcrQuery]:
    """Read the gold set committed before either engine ran."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [OcrQuery(**row) for row in rows]


def fold(text: str, keep_spaces: bool = True) -> str:
    """Case-, accent- and whitespace-insensitive form for substring matching.

    Accents are folded because the gold strings were typed unaccented; that
    loosening applies to every engine equally. Spaces are preserved by default
    because losing them is a real defect, not a formatting difference.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    stripped = stripped.replace("'", "'").casefold()
    collapsed = _WHITESPACE.sub(" ", stripped).strip()
    return collapsed if keep_spaces else collapsed.replace(" ", "")


def _page_text(records: list[Any], document_name: str, page_number: int) -> str:
    """The recognised text for one page, empty when it produced none."""
    for record in records:
        if record.document_name == document_name and record.page_number == page_number:
            return str(record.text)
    return ""


def _rank_of_gold(results: list[Any], query: OcrQuery) -> int | None:
    """1-based rank of the first chunk from the gold page, or None."""
    for position, result in enumerate(results, start=1):
        if result.document_name == query.document_name and result.page_number == query.gold_page:
            return position
    return None


def evaluate(engine_name: str, input_dir: Path, queries: list[OcrQuery], work_dir: Path) -> dict[str, Any]:
    """Ingest the whole corpus under one OCR setting and score the gold set."""
    from pdf_search.pdf_text import discover_pdfs, extract_pages

    engine = build_engine(engine_name)
    snapshot = work_dir / "storage"

    started = time.monotonic()
    records = [rec for pdf in discover_pdfs(input_dir) for rec in extract_pages(pdf, engine)]
    extraction_seconds = time.monotonic() - started

    embedder = SentenceTransformerEmbedder()
    started = time.monotonic()
    manifest = run_ingestion(input_dir, snapshot, embedder, ocr_engine=engine)
    ingestion_seconds = time.monotonic() - started

    loaded = storage.load(snapshot)
    strict = loose = 0
    hits_string = hits_question = 0
    ranks: list[int] = []

    for query in queries:
        page_text = _page_text(records, query.document_name, query.gold_page)
        if fold(query.gold_string) in fold(page_text):
            strict += 1
        if fold(query.gold_string, keep_spaces=False) in fold(page_text, keep_spaces=False):
            loose += 1

        for text, counter in ((query.gold_string, "string"), (query.tier_b_query, "question")):
            results = loaded.search(embedder.encode_query(text), SEARCH_TOP_K)
            rank = _rank_of_gold(results, query)
            if rank is not None and rank <= PRIMARY_K:
                if counter == "string":
                    hits_string += 1
                else:
                    hits_question += 1
            if rank is not None and counter == "string":
                ranks.append(rank)

    scanned = queries[0].document_name
    regression = _existing_corpus_unchanged(loaded, embedder, scanned)

    return {
        "engine": engine_name or "none",
        "n_gold": len(queries),
        "fidelity_strict": strict,
        "fidelity_ignoring_spaces": loose,
        f"retrieval_string_hits@{PRIMARY_K}": hits_string,
        f"retrieval_question_hits@{PRIMARY_K}": hits_question,
        "median_rank_when_found": statistics.median(ranks) if ranks else None,
        "n_chunks": manifest.n_chunks,
        "extraction_seconds": round(extraction_seconds, 2),
        "ingestion_seconds": round(ingestion_seconds, 2),
        **regression,
    }


def _existing_corpus_unchanged(
    loaded: Any, embedder: Any, scanned_document: str
) -> dict[str, Any]:
    """Check the clause that OCR must be strictly additive.

    Chunk *indices* necessarily shift, because the scanned document sorts third
    of seven and its chunks are inserted rather than appended -- that is exactly
    what adding a document to a corpus does. What must not change is the text of
    every chunk that already existed, and what the existing labelled set returns.
    So the digest is taken over (document, page, text) of the chunks that do not
    come from the scanned document, and the 26 embedding queries are re-scored.
    """
    payload = "\n".join(
        f"{c.document_name}|{c.page_number}|{c.text}"
        for c in loaded.chunks
        if c.document_name != scanned_document
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    hits = {"A": 0, "B": 0}
    for query in load_embedding_queries(Path("eval/queries.jsonl")):
        results = loaded.search(embedder.encode_query(query.query), SEARCH_TOP_K)
        rank = embedding_rank(results, query)
        if rank is not None and rank <= PRIMARY_K:
            hits[query.tier] += 1

    return {
        "pre_existing_chunks": sum(1 for c in loaded.chunks if c.document_name != scanned_document),
        "pre_existing_chunks_sha256": digest,
        "embedding_tier_a_hits@5": hits["A"],
        "embedding_tier_b_hits@5": hits["B"],
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Define the CLI."""
    parser = argparse.ArgumentParser(description="Score one OCR setting on the gold set.")
    parser.add_argument("--engine", default="", help="rapidocr, tesseract, or empty for none")
    parser.add_argument("--queries", type=Path, default=Path("eval/ocr_queries.jsonl"))
    parser.add_argument("--input-dir", type=Path, default=Path("sample-pdfs"))
    parser.add_argument("--out-dir", type=Path, default=Path("eval/results"))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Evaluate one OCR setting and persist the result immediately."""
    args = build_argument_parser().parse_args(argv)
    queries = load_queries(args.queries)

    with tempfile.TemporaryDirectory(prefix="pdf-search-ocr-") as raw:
        result = evaluate(args.engine, args.input_dir, queries, Path(raw))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"ocr--{result['engine']}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    for key, value in result.items():
        print(f"  {key:32} {value}")
    print(f"  written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
