# Decision rule: should scanned pages be OCR'd, and by which engine?

Written and committed **before either engine was run**. The commit order is the
evidence, as it was for the embedding decision in [`DECISION.md`](DECISION.md).

## The question

One of the seven supplied documents,
`AFF-2026.06.11-DP-26A0019-19-PLACE-DES-HAUTS-TAILLIS-ACCORD.pdf`, has no text
layer on either of its two pages. `pypdfium2` extracts nothing, the chunker
receives nothing, and the document contributes **0 chunks**. It is not
retrievable at any _k_ by any query, and no better embedding model changes that
— the content never reaches the index.

The pipeline already detects the condition and reports it (`[no_text]`, plus a
warning naming documents that contributed nothing), so routing is not the hard
part: **only pages that yield no text are candidates for OCR**, which bounds the
cost to 2 pages of 64 by construction. `pdf_text` is one of the three seams the
README names, put there for exactly this.

The open questions are whether OCR output is good enough to be worth indexing,
which engine to use, and whether adding one is worth what it costs the image.

## Candidates

| Engine                              | Install   | Weights            | System dependency      |
| ----------------------------------- | --------- | ------------------ | ---------------------- |
| `rapidocr-onnxruntime`              | pip       | ~15 MB ONNX        | none                   |
| `pytesseract` + `tesseract-ocr-fra` | pip + apt | ~15 MB traineddata | `tesseract-ocr` binary |

Both are open-weight, CPU-only and offline once installed, which the container
requires. Excluded before running: **docTR** and **Surya** add a second deep
model to an image already at 2.02 GB, and any VLM-based OCR is far outside a
CPU-only budget.

## How quality is measured

Two metrics, because they answer different questions and can disagree.

**Fidelity.** A set of strings read off the two rendered pages by eye and
recorded in `eval/ocr_queries.jsonl` before either engine ran. An engine scores
a hit when the string appears in its output for that page after the same
normalisation the ingestion pipeline applies. This measures OCR directly.

**Retrievability — the metric that decides.** The OCR'd pages are chunked,
embedded and indexed exactly like every other page, and each gold string is
issued as a query. A query is a hit at _k_ if any of the top-_k_ chunks comes
from the page the string was read from. The baseline is **0 by construction**:
the pages are absent from the index today. This measures the only thing that
matters — whether the document became findable.

Fidelity can be high while retrievability is low (a page recognised well but
embedded poorly), and retrievability can be high while fidelity is imperfect (a
mangled accent that still leaves enough signal). Both are reported.

**These labels are weaker evidence than the embedding set's.** There the gold
was structural — the page a verbatim string was physically lifted from. Here I
read the strings off a scan myself, so the labels are authored, and there are
only 2 pages. No conclusion below is a general claim about either engine; it is
a claim about these two pages.

**The gold set contains no personal data.** The scan names a private individual
and their address. That is a lawfully public _affichage légal_ and the PDF is
published, but this file is tracked in git and plain text, so every gold string
is drawn from the impersonal parts of the document — the dossier number, the
cadastral reference, the legal headings, the nature of the works.

## The decision rule

> **Ship OCR only if the better engine makes at least 75% of the gold strings
> retrievable at k=5** on pages that are currently unretrievable at any _k_.
>
> And it must cost no more than **300 MB** added to the built image, and no more
> than **60 s** added to a full ingestion of the seven documents.
>
> And it must be **strictly additive**: all 387 chunks that exist today must
> remain byte-identical, and every one of the 26 queries in
> [`queries.jsonl`](queries.jsonl) must return exactly what it returns now. OCR
> may only touch pages that yield no text. A regression on the text pages fails
> the rule outright, whatever the OCR quality.
>
> If both engines clear the bar within **2 gold strings** of each other, take the
> one with **no system dependency**, because an apt layer is a real cost to a
> reviewer told the project runs in a few commands.
>
> If neither clears the bar, **ship no OCR and publish the table showing why.**
> The current behaviour — extract nothing, say so loudly, name the document in a
> warning — is a defensible answer, and a wrong answer indexed as though it were
> right is worse than a documented gap.

**Why 75%.** With 16 gold strings the resolution is about one string in six.
Requiring near-perfection would fail an engine for a single mangled accent;
requiring a bare majority would ship something that half-works. 75% says most of
the page is recoverable without pretending the number is precise.

## Amendment, 2026-09-02, before either engine was scored

Two clarifications, both settled before any result existed. Neither moves the
bar; they make an underspecified rule executable.

**Fidelity matching is accent- and case-insensitive.** The rule said a gold
string counts when it appears "after the same normalisation the ingestion
pipeline applies". That normalisation is NFKC, which preserves accents -- but
the gold strings in `ocr_queries.jsonl` were typed unaccented, so under a literal
reading almost nothing could ever match. Matching therefore folds accents and
case. It is a loosening, and it applies identically to both engines.

**Fidelity is reported twice: with spaces respected, and ignoring them.** A
recogniser whose character set has no space token emits text that a human can
still read and that tokenises into nonsense, which is invisible to any metric
that normalises whitespace away. Reporting only the space-insensitive number
would conceal exactly the failure that destroys retrieval; reporting only the
strict number would look like a formatting quibble. Both are published, and the
rule is applied to retrievability regardless, which is unaffected by this choice.

**Each engine is evaluated as it installs.** `rapidocr-onnxruntime` from pip and
`tesseract-ocr` plus `tesseract-ocr-fra` from apt. Neither is given a
hand-picked alternative model, because the thing being compared is what a
reviewer would actually get from the documented install. Where a default turns
out to be a poor fit for French, that is a finding about the candidate rather
than a handicap to be corrected.

**Both engines run in the same Linux container** (`eval/Dockerfile.ocr-eval`),
because one needs a system package and the other does not, and a comparison
split across two operating systems would measure the machines.

## What this evaluation is not

It is not a general OCR benchmark. It is 2 pages of one clean, high-resolution
scan of a printed French administrative form. It says nothing about handwriting,
skew, low-resolution faxes, or multi-column layouts, all of which are common in
this document class and none of which are represented here.
