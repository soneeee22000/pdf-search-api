"""Command-line ingestion: PDFs in, a persisted searchable index out.

Every run is a full rebuild. At this corpus size a rebuild costs seconds and it
removes the entire class of drift bugs that a partial re-ingest path creates.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pdf_search import storage
from pdf_search.chunking import CHUNK_TOKEN_BUDGET, chunk_page
from pdf_search.embeddings import DEFAULT_MODEL_NAME, Embedder
from pdf_search.pdf_text import discover_pdfs, extract_pages
from pdf_search.schemas import ChunkRecord, Manifest, PageRecord

logger = logging.getLogger(__name__)


def build_argument_parser() -> argparse.ArgumentParser:
    """Define the CLI. The PDF folder path is a required argument."""
    parser = argparse.ArgumentParser(
        prog="python -m pdf_search.ingest",
        description="Extract, chunk, embed and index a folder of PDF documents.",
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True, help="folder containing the PDF files to ingest"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("storage"), help="where to write the index snapshot"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_NAME, help="sentence-transformers model identifier"
    )
    parser.add_argument("--verbose", action="store_true", help="log per-document detail")
    return parser


def collect_chunks(pages: list[PageRecord], embedder: Embedder) -> list[ChunkRecord]:
    """Chunk every extracted page, numbering chunks contiguously across the corpus."""
    chunks: list[ChunkRecord] = []
    for page in pages:
        chunks.extend(chunk_page(page, embedder.count_tokens, start_index=len(chunks)))
    return chunks


def run_ingestion(input_dir: Path, output_dir: Path, embedder: Embedder) -> Manifest:
    """Extract, chunk, embed and persist. Returns the manifest that was written."""
    started = time.monotonic()
    pdf_paths = discover_pdfs(input_dir)
    logger.info("found %d PDF file(s) in %s", len(pdf_paths), input_dir)

    pages: list[PageRecord] = []
    for pdf_path in pdf_paths:
        document_pages = extract_pages(pdf_path)
        pages.extend(document_pages)
        _log_document(document_pages, pdf_path.name)

    chunks = collect_chunks(pages, embedder)
    if not chunks:
        raise RuntimeError(
            "no text could be extracted from any document, so there is nothing to index. "
            "The PDFs may be scanned images, which need OCR."
        )

    vectors = embedder.encode([chunk.text for chunk in chunks])
    index = storage.build_index(vectors, embedder.dim)

    manifest = Manifest(
        model_name=embedder.name,
        embedding_dim=embedder.dim,
        chunk_token_budget=CHUNK_TOKEN_BUDGET,
        n_documents=len(pdf_paths),
        n_chunks=len(chunks),
        n_pages=len(pages),
        n_pages_no_text=sum(1 for p in pages if p.status != "extracted"),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source_documents=[p.name for p in pdf_paths],
    )

    storage.save(output_dir, index, chunks, manifest)
    _print_summary(manifest, pages, chunks, input_dir, output_dir, time.monotonic() - started)
    return manifest


def _log_document(pages: list[PageRecord], name: str) -> None:
    """Report per-document extraction outcome at debug level."""
    counts = Counter(page.status for page in pages)
    logger.debug(
        "%s: %d page(s) %s", name, len(pages), ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )


def _print_summary(
    manifest: Manifest,
    pages: list[PageRecord],
    chunks: list[ChunkRecord],
    input_dir: Path,
    output_dir: Path,
    elapsed: float,
) -> None:
    """Print what was ingested, including the pages that yielded nothing.

    Pages with no text layer are named rather than counted silently: a corpus
    where a document contributed nothing must not look like a successful run.
    """
    print(f"\nIngested {manifest.n_documents} document(s) from {input_dir}")
    print(f"  pages           : {manifest.n_pages}")
    print(f"  pages with text : {manifest.n_pages - manifest.n_pages_no_text}")
    print(f"  pages no text   : {manifest.n_pages_no_text}")
    print(f"  chunks          : {manifest.n_chunks}")
    print(f"  characters      : {sum(len(c.text) for c in chunks):,}")
    print(f"  model           : {manifest.model_name} ({manifest.embedding_dim}-d)")
    print(f"  elapsed         : {elapsed:.1f}s")
    print(f"  index written to: {output_dir}")

    unusable = [p for p in pages if p.status != "extracted"]
    if unusable:
        print("\n  Pages that yielded no usable text (candidates for OCR):")
        for page in unusable:
            print(f"    {page.document_name} p.{page.page_number} [{page.status}]")

    empty_documents = sorted(
        {p.document_name for p in pages}
        - {p.document_name for p in pages if p.status == "extracted"}
    )
    if empty_documents:
        print("\n  WARNING - these documents contributed nothing to the index:")
        for name in empty_documents:
            print(f"    {name}")
    print()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_argument_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )

    from pdf_search.embeddings import SentenceTransformerEmbedder

    try:
        embedder = SentenceTransformerEmbedder(args.model)
        run_ingestion(args.input_dir, args.output_dir, embedder)
    except (NotADirectoryError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
