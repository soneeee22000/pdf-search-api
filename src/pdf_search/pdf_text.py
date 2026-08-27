"""Per-page text extraction and the normalisation the embedding model needs.

Extraction uses pypdfium2 (Apache-2.0/BSD-3) rather than PyMuPDF, to avoid an
AGPL/commercial dependency in a proprietary-service context.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import pypdfium2

from pdf_search.schemas import PageRecord

logger = logging.getLogger(__name__)

SOFT_HYPHEN = "­"
RIGHT_SINGLE_QUOTE = "’"

# A page yielding fewer than this many non-space characters is treated as
# having no usable text layer. Deliberately low: the point is to flag pages an
# operator should look at, not to make a fine-grained judgement about why.
MIN_USABLE_CHARS = 20

_CARRIAGE_RETURN = re.compile(r"\r\n?")
_LINE_WRAP_HYPHEN = re.compile(r"([a-zà-ÿ])-\n([a-zà-ÿ])")
_WHITESPACE_RUN = re.compile(r"[ \t ]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def normalise(text: str) -> str:
    """Apply the text hygiene that measurably affects embedding quality.

    NFKC folds the no-break and narrow-no-break spaces that French typography
    mandates before ':', ';', '?' and '!', and resolves the fi/fl ligatures.
    It does not touch the soft hyphen or the typographic apostrophe, so both
    are handled explicitly.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _CARRIAGE_RETURN.sub("\n", text)
    text = text.replace(SOFT_HYPHEN, "")
    text = text.replace(RIGHT_SINGLE_QUOTE, "'")
    text = _LINE_WRAP_HYPHEN.sub(r"\1\2", text)
    text = _WHITESPACE_RUN.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def _page_status(text: str) -> str:
    """Classify a page by whether it produced usable text."""
    return "extracted" if len(text.replace(" ", "")) >= MIN_USABLE_CHARS else "no_text"


def extract_pages(pdf_path: Path) -> list[PageRecord]:
    """Extract one `PageRecord` per page, in reading order, 1-based.

    A page that cannot be read yields a record with status 'error' rather than
    aborting the document, and a document that cannot be opened yields a single
    error record rather than aborting the run.
    """
    document_name = pdf_path.name
    try:
        document = pypdfium2.PdfDocument(str(pdf_path))
    except Exception as exc:  # noqa: BLE001 - a corrupt file must not stop the corpus
        logger.warning("cannot open %s: %s", document_name, exc)
        return [PageRecord(document_name=document_name, page_number=1, status="error", char_count=0)]

    records: list[PageRecord] = []
    try:
        for page_number, page in enumerate(document, start=1):
            records.append(_extract_one_page(page, document_name, page_number))
    finally:
        document.close()
    return records


def _extract_one_page(page: object, document_name: str, page_number: int) -> PageRecord:
    """Extract a single page, converting any failure into an 'error' record."""
    try:
        text_page = page.get_textpage()  # type: ignore[attr-defined]
        try:
            raw = text_page.get_text_range()
        finally:
            text_page.close()
    except Exception as exc:  # noqa: BLE001 - one unreadable page must not stop the document
        logger.warning("cannot read %s page %d: %s", document_name, page_number, exc)
        return PageRecord(
            document_name=document_name, page_number=page_number, status="error", char_count=0
        )

    text = normalise(raw)
    return PageRecord(
        document_name=document_name,
        page_number=page_number,
        status=_page_status(text),
        char_count=len(text),
        text=text,
    )


def discover_pdfs(input_dir: Path) -> list[Path]:
    """List the PDFs to ingest in a deterministic, platform-independent order.

    Sorting `Path` objects directly is not portable: on Windows the comparison
    is case-insensitive, on Linux it is not, so the same corpus would produce
    different chunk indices on a developer machine and in the Docker image.
    Sorting on an explicit case-folded key removes that difference.
    """
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input directory does not exist: {input_dir}")
    pdfs = sorted(
        (p for p in input_dir.iterdir() if p.suffix.lower() == ".pdf" and p.is_file()),
        key=lambda path: (path.name.casefold(), path.name),
    )
    if not pdfs:
        raise FileNotFoundError(f"no PDF files found in {input_dir}")
    return pdfs
