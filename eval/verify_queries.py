"""Check that the evaluation labels are structural, not graded.

The claim this repository makes about its evaluation set is that relevance was
never a judgement call: each gold label is the page a string was physically
lifted from. This script is what makes that claim checkable rather than asserted.

For every item it verifies that

1. the Tier A string appears **verbatim** on the page it claims, and
2. it appears on **no other page** of that document, so the label is unambiguous,
3. and it reports any distinctive word Tier B still shares with Tier A, since a
   paraphrase that reuses the rare tokens stops discriminating between models.

Comparison is accent- and case-insensitive with whitespace collapsed, because
that is the only difference extraction can legitimately introduce.

    python eval/verify_queries.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pdf_search.pdf_text import discover_pdfs, extract_pages  # noqa: E402

# Shorter words are common French function words; flagging them would be noise.
DISTINCTIVE_MIN_LENGTH = 5


def fold(text: str) -> str:
    """Accent- and case-insensitive form with whitespace collapsed."""
    stripped = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"\s+", " ", stripped.lower()).strip()


def distinctive_words(text: str) -> set[str]:
    """Words long enough to carry retrieval signal."""
    return {w for w in re.findall(r"[a-z0-9]+", fold(text)) if len(w) >= DISTINCTIVE_MIN_LENGTH}


def load_pages(input_dir: Path) -> dict[tuple[str, int], str]:
    """Extracted text of every page that has any, keyed by document and page."""
    pages: dict[tuple[str, int], str] = {}
    for pdf_path in discover_pdfs(input_dir):
        for page in extract_pages(pdf_path):
            if page.status == "extracted":
                pages[(pdf_path.name, page.page_number)] = page.text
    return pages


def check_item(item: dict[str, object], pages: dict[tuple[str, int], str]) -> list[str]:
    """Return the problems with one item; empty means it is sound."""
    document = str(item["document_name"])
    gold_page = int(str(item["gold_page"]))
    tier_a = str(item["tier_a_query"])
    problems: list[str] = []

    page_text = pages.get((document, gold_page))
    if page_text is None:
        return [f"gold page {gold_page} of {document} has no extracted text"]

    needle = fold(tier_a)
    if needle not in fold(page_text):
        problems.append("Tier A does not appear verbatim on its gold page")

    elsewhere = sorted(
        page
        for (doc, page), text in pages.items()
        if doc == document and page != gold_page and needle in fold(text)
    )
    if elsewhere:
        problems.append(f"Tier A also appears on page(s) {elsewhere}, so the label is ambiguous")

    shared = distinctive_words(tier_a) & distinctive_words(str(item["tier_b_query"]))
    if shared:
        problems.append(f"Tier B reuses {sorted(shared)}")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Verify every item, printing a line per problem. Non-zero on any failure."""
    parser = argparse.ArgumentParser(description="Verify the evaluation labels are structural.")
    parser.add_argument("--queries", type=Path, default=Path("eval/queries.jsonl"))
    parser.add_argument("--input-dir", type=Path, default=Path("sample-pdfs"))
    args = parser.parse_args(argv)

    pages = load_pages(args.input_dir)
    items = [
        json.loads(line)
        for line in args.queries.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    failures = 0
    for item in items:
        for problem in check_item(item, pages):
            marker = "SHARED " if problem.startswith("Tier B reuses") else "FAIL   "
            if marker == "FAIL   ":
                failures += 1
            print(f"{marker}p{item['gold_page']:<3} {str(item['tier_a_query'])[:44]:<46} {problem}")

    gold_pages = {(i["document_name"], i["gold_page"]) for i in items}
    print(
        f"\n{len(items)} items over {len(gold_pages)} distinct gold pages; "
        f"{failures} verbatim or uniqueness failure(s)"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
