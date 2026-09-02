"""Run one embedding model end to end over the corpus and score it.

One model per invocation, one JSON result file out. Each candidate is ingested
with its own token budget and therefore its own chunk set, which is why scoring
is done at page level: chunk indices are not comparable across models, pages are.

Quality is only half of it. A model that wins by two queries and triples the
image is not a win, so the operational axes are measured in the same pass.

Start with --smoke. It runs the whole path on two documents and three queries in
a couple of minutes and surfaces configuration errors before an hour of sweep
does.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pdf_search import storage  # noqa: E402
from pdf_search.embeddings import (  # noqa: E402
    SentenceTransformerEmbedder,
    chunk_budget_for,
)
from pdf_search.ingest import run_ingestion  # noqa: E402

RECALL_AT = (1, 3, 5, 10)
MRR_DEPTH = 10
SEARCH_TOP_K = 10
SMOKE_DOCUMENTS = 2
SMOKE_QUERIES = 3


@dataclass(frozen=True)
class EvalQuery:
    """One labelled query. The gold label is a page, never a chunk."""

    tier: str
    query: str
    document_name: str
    gold_page: int


def load_queries(path: Path) -> list[EvalQuery]:
    """Read the two-tier query set, flattening each item into its two tiers."""
    queries: list[EvalQuery] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for tier, key in (("A", "tier_a_query"), ("B", "tier_b_query")):
            queries.append(
                EvalQuery(
                    tier=tier,
                    query=row[key],
                    document_name=row["document_name"],
                    gold_page=int(row["gold_page"]),
                )
            )
    return queries


def _hit_rank(results: list[Any], query: EvalQuery) -> int | None:
    """1-based rank of the first result from the gold page, or None."""
    for rank, result in enumerate(results, start=1):
        if result.document_name == query.document_name and result.page_number == query.gold_page:
            return rank
    return None


def score_tier(ranks: list[int | None]) -> dict[str, Any]:
    """Page-level recall@k and MRR, reported as counts as well as rates."""
    total = len(ranks)
    if total == 0:
        return {"n": 0}
    scored: dict[str, Any] = {"n": total}
    for k in RECALL_AT:
        hits = sum(1 for rank in ranks if rank is not None and rank <= k)
        scored[f"recall@{k}"] = round(hits / total, 4)
        scored[f"hits@{k}"] = hits
    reciprocal = [1.0 / r for r in ranks if r is not None and r <= MRR_DEPTH]
    scored[f"mrr@{MRR_DEPTH}"] = round(sum(reciprocal) / total, 4)
    return scored


def _model_disk_bytes(model_name: str) -> int | None:
    """Size of the model in the HuggingFace cache, or None if not locatable."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return None
    folder = Path(HF_HUB_CACHE) / f"models--{model_name.replace('/', '--')}"
    if not folder.exists():
        return None
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())


def _peak_rss_bytes() -> int | None:
    """Peak resident set size of this process, if the platform reports it."""
    try:
        import psutil
    except ImportError:
        return None
    info = psutil.Process().memory_info()
    return int(getattr(info, "peak_wset", info.rss))


def _smoke_corpus(input_dir: Path, work_dir: Path) -> Path:
    """A two-document copy of the corpus, so the smoke path stays fast."""
    target = work_dir / "smoke-input"
    target.mkdir(parents=True, exist_ok=True)
    for pdf in sorted(input_dir.glob("*.pdf"))[:SMOKE_DOCUMENTS]:
        shutil.copy2(pdf, target / pdf.name)
    return target


def evaluate(
    model_name: str,
    input_dir: Path,
    queries: list[EvalQuery],
    work_dir: Path,
    budget_override: int | None = None,
) -> dict[str, Any]:
    """Ingest the corpus under one model, then score every query against it.

    `budget_override` exists for one purpose: giving two models the same chunk
    size so the comparison isolates the model. By default each model gets the
    budget its own window earns, which is the deployment-realistic setting but
    confounds model with chunk size.
    """
    embedder = SentenceTransformerEmbedder(model_name)
    budget = budget_override or chunk_budget_for(model_name, embedder.max_seq_length)
    snapshot = work_dir / "storage"

    started = time.monotonic()
    manifest = run_ingestion(input_dir, snapshot, embedder, budget)
    ingestion_seconds = time.monotonic() - started

    loaded = storage.load(snapshot)
    ranks: list[int | None] = []
    embed_ms: list[float] = []
    search_ms: list[float] = []

    for query in queries:
        mark = time.perf_counter()
        vector = embedder.encode_query(query.query)
        embed_ms.append((time.perf_counter() - mark) * 1000)

        mark = time.perf_counter()
        results = loaded.search(vector, SEARCH_TOP_K)
        search_ms.append((time.perf_counter() - mark) * 1000)

        ranks.append(_hit_rank(results, query))

    # Kept parallel to `queries` rather than bucketed, so a per-query row can
    # never be paired with another query's rank.
    by_tier = {
        tier: [r for q, r in zip(queries, ranks, strict=True) if q.tier == tier]
        for tier in ("A", "B")
    }

    return {
        "model_name": model_name,
        "embedding_dim": embedder.dim,
        "max_seq_length": embedder.max_seq_length,
        "chunk_token_budget": budget,
        "query_prefix": embedder.prefixes.query,
        "document_prefix": embedder.prefixes.document,
        "n_chunks": manifest.n_chunks,
        "n_pages": manifest.n_pages,
        "ingestion_seconds": round(ingestion_seconds, 2),
        "index_bytes": (snapshot / "index.faiss").stat().st_size,
        "model_disk_bytes": _model_disk_bytes(model_name),
        "peak_rss_bytes": _peak_rss_bytes(),
        "latency_ms": _latency_summary(embed_ms, search_ms),
        "tier_a": score_tier(by_tier["A"]),
        "tier_b": score_tier(by_tier["B"]),
        "per_query": [
            {**asdict(q), "rank": r} for q, r in zip(queries, ranks, strict=True)
        ],
    }


def _latency_summary(embed_ms: list[float], search_ms: list[float]) -> dict[str, float]:
    """Median embed and search cost, which is the whole index-choice argument."""
    if not embed_ms:
        return {}
    return {
        "embed_p50": round(statistics.median(embed_ms), 3),
        "embed_max": round(max(embed_ms), 3),
        "search_p50": round(statistics.median(search_ms), 3),
        "search_max": round(max(search_ms), 3),
        "embed_over_search": round(statistics.median(embed_ms) / statistics.median(search_ms), 1),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    """Define the CLI."""
    parser = argparse.ArgumentParser(description="Score one embedding model on the labelled set.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--queries", type=Path, default=Path("eval/queries.jsonl"))
    parser.add_argument("--input-dir", type=Path, default=Path("sample-pdfs"))
    parser.add_argument("--out-dir", type=Path, default=Path("eval/results"))
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="force a chunk token budget, to isolate the model from the chunk size",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="two documents, three queries: proves the path before the sweep",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Evaluate one model and persist its result immediately."""
    args = build_argument_parser().parse_args(argv)
    queries = load_queries(args.queries)

    with tempfile.TemporaryDirectory(prefix="pdf-search-eval-") as raw:
        work_dir = Path(raw)
        input_dir = args.input_dir
        if args.smoke:
            input_dir = _smoke_corpus(args.input_dir, work_dir)
            queries = queries[:SMOKE_QUERIES]
        result = evaluate(args.model, input_dir, queries, work_dir, args.budget)

    result["smoke"] = args.smoke
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "--smoke" if args.smoke else ""
    if args.budget is not None:
        suffix += f"--budget{args.budget}"
    slug = args.model.replace("/", "__") + suffix
    out_path = args.out_dir / f"{slug}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{args.model}")
    print(f"  chunks {result['n_chunks']}  budget {result['chunk_token_budget']}  "
          f"dim {result['embedding_dim']}  ingest {result['ingestion_seconds']}s")
    print(f"  tier A {result['tier_a']}")
    print(f"  tier B {result['tier_b']}")
    print(f"  latency {result['latency_ms']}")
    print(f"  written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
