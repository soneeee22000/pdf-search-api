# Model selection: the rule, written before the results

This file is committed **before any evaluation has been run**. Its commit
timestamp is the evidence. The point is that the conclusion cannot be fitted to
the numbers after seeing them, which is the failure mode that makes most
self-run model comparisons worthless.

## The question

The brief suggests `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
and notes that the suggestion is only an example. The rest of this repository
argues its choices — pypdfium2 over PyMuPDF on licence, `IndexFlatIP` over
`IndexFlatL2` on score semantics, FAISS over a bare numpy matmul on persistence
— but it accepted the embedding model as given and designed around the
consequences. This is the missing decision.

Two properties of the incumbent motivate the question:

1. **It is a paraphrase/STS model, not a retrieval model.** It is a multilingual
   knowledge-distillation student of an *English paraphrase teacher* (Reimers &
   Gurevych, EMNLP 2020), trained to mimic that teacher's geometry on translated
   sentence pairs. It has never seen a query-to-passage contrastive objective,
   and MTEB records `use_instructions: false` for it — it has no notion of a
   query role at all. Sentence-Transformers' own documentation files it under
   *similarity* models and points elsewhere for search.
2. **Its 128-token window caps the chunking strategy.** Measured on this corpus:
   nothing is truncated, because the chunker enforces the budget — but **59 of
   the 62 pages with text are split into more than one chunk**, only **3 of 62**
   fit in a single chunk, and one page fragments into 14. At a 512-token window
   **23 of 62** would fit whole. The cost is fragmentation — an article
   separated from the citation that gives it meaning — not truncation.

## Candidates

| Model | Dim | Window | Params | Camp | Prefixes |
|---|---|---|---|---|---|
| `paraphrase-multilingual-MiniLM-L12-v2` *(incumbent)* | 384 | 128 | 118M | paraphrase / STS | none |
| `intfloat/multilingual-e5-small` | 384 | 512 | 118M | retrieval | `query: ` / `passage: ` |
| `intfloat/multilingual-e5-base` | 768 | 512 | 278M | retrieval | `query: ` / `passage: ` |
| `ibm-granite/granite-embedding-97m-multilingual-r2` | 384 | 32k | 97M | retrieval | none documented |
| `OrdalieTech/Solon-embeddings-base-0.1` | 768 | 514 | ~300M | retrieval, French-specialised | `query : ` on the query only |

`multilingual-e5-small` is the interesting one: same 384 width, same ~118M
parameters, same backbone family (`microsoft/Multilingual-MiniLM-L12-H384`), so
the index width, storage layout and image size are all unchanged — and yet it is
retrieval-trained with a 512-token window.

**Excluded, with reasons:**

- `BAAI/bge-m3` — 2.27 GB of weights would dominate an image the brief asks to be
  runnable in a few commands, at roughly 5x the CPU encode cost. Its combined
  dense+sparse mode is genuinely attractive and is noted as future work instead.
- `jinaai/jina-embeddings-v3` — CC BY-NC 4.0. Non-commercial licences are
  disqualifying for a service, regardless of benchmark position.
- `google/embeddinggemma-300m` — its Matryoshka checkpoints are 768/512/256/128;
  384 is not offered, so it cannot be a drop-in.
- `Lajavaness/bilingual-embedding-small` — 384-d and French-capable, but
  explicitly fine-tuned for STS. Same camp as the incumbent, so it does not test
  the hypothesis.

## How quality is measured

**Labels are structural. Only the phrasings are authored.** Relevance is never a
judgement call, because the gold label is not graded — it is the page a string
was physically lifted from.

- **Tier A — known-item retrieval.** The query is a verbatim structural string
  from the corpus: a document title, an article heading, an agenda line, a table
  row label. Gold = the page it came from.
  *Declared bias:* the query shares rare tokens with its target, which favours
  lexical matching and understates dense models. It is one axis, not an accuracy
  score.
- **Tier B — paraphrased queries.** A natural French question for the same gold
  page that **deliberately avoids the distinctive tokens** of the Tier A string.
  The label carries over unchanged, so it is still structural.
  **This is the tier that discriminates between embedding models, and it is the
  primary metric.**

The two tiers are **reported separately and never merged.**

**Metrics are computed at page level, never chunk level.** Candidates have
different token budgets and therefore different chunk sets; chunk-level labels
would not be comparable across models. A query counts as a hit at *k* if any of
the top-*k* chunks comes from the gold page.

Reported: recall@1, @3, @5, @10 and MRR@10, **as counts alongside percentages**,
with *n* stated every time.

## The decision rule

> **Switch away from the incumbent only if a candidate wins Tier B page-level
> recall@5 by more than two queries**, and costs no more than **2x** incumbent
> ingestion wall-clock, no more than **2x** peak RSS, and ships weights of
> **1.5 GB or less**.
>
> If several candidates clear the bar within two queries of each other, take the
> cheapest by parameter count.
>
> If the incumbent is within two queries of the best candidate, **keep the
> incumbent** — the status-quo bias is deliberate, because switching costs a
> re-index and the incumbent is what the brief suggested.
>
> If the incumbent holds, **it stays, and the table showing why is published.**
> That is a result, not a failed experiment.

**Why two queries.** At n≈25 with recall near 0.7, the standard error is about
sqrt(0.7 x 0.3 / 25) ≈ 0.09, or roughly 2.3 queries. A margin of one or two
queries is inside the sampling noise of a set this size. This set cannot resolve
differences finer than that, and no result below is reported as if it could.

## Amendment, 2026-09-02, before any evaluation was run

`ibm-granite/granite-embedding-97m-multilingual-r2` is **withdrawn from the
candidate list**. It does not load:

    ValueError: The checkpoint you are trying to load has model type `modernbert`
    but Transformers does not recognize this architecture.

The exclusion is mechanical rather than a judgement about the model: it needs a
newer `transformers` than this project pins, and raising a core dependency to
admit one candidate is a larger change than the candidate is worth here. The
dependency cost is itself the finding.

This is recorded as an amendment rather than as an edit to the list above, and it
is committed before any result exists. The remaining candidates are the
incumbent, `multilingual-e5-small`, `multilingual-e5-base` and
`Solon-embeddings-base-0.1`.

Confirmed on load: e5-small is **384-d with a 512-token window**, e5-base is
768-d/512, Solon-base is 768-d/512.

## What this evaluation is not

- It is **not** a benchmark. It is one corpus, ~25 queries, and the phrasings are
  mine. A larger labelled set is listed as future work.
- It **cannot** rank models in general. It can only answer whether a candidate is
  clearly better *on these seven documents*.
- It measures **dense retrieval only.** The known failure on this corpus — an
  exact document title returning its own document third — is lexical, and the
  fix for it is a BM25 lane, not a different embedding model. No model here is
  expected to fix that, and if one appears to, that is a reason for suspicion
  rather than celebration.
