"""Build the comparison table and apply the pre-registered decision rule.

The rule from DECISION.md is executed here rather than eyeballed, so the verdict
follows from the numbers mechanically. The thresholds below must match that file;
they were fixed before any result existed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

INCUMBENT = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Thresholds, pre-registered in DECISION.md. Do not tune these to a result.
PRIMARY_METRIC = "hits@5"
MARGIN_QUERIES = 2
MAX_INGEST_RATIO = 2.0
MAX_RSS_RATIO = 2.0
MAX_MODEL_BYTES = 1.5 * 1024**3

BYTES_PER_MB = 1024 * 1024


def load_results(directory: Path) -> list[dict[str, Any]]:
    """Read every non-smoke result, incumbent first."""
    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
        if "--smoke" not in path.name
    ]
    return sorted(results, key=lambda r: r["model_name"] != INCUMBENT)


def _mb(value: int | None) -> str:
    """Format a byte count in MB, or a dash when the platform did not report it."""
    return "-" if value is None else f"{value / BYTES_PER_MB:.0f}"


def quality_table(results: list[dict[str, Any]]) -> str:
    """Tier A and Tier B side by side, counts alongside rates."""
    n_a = results[0]["tier_a"]["n"]
    n_b = results[0]["tier_b"]["n"]
    lines = [
        f"| Model | Dim | Window | Budget | Chunks | "
        f"Tier A r@5 (n={n_a}) | Tier B r@5 (n={n_b}) | Tier B MRR@10 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in results:
        mark = " *(incumbent)*" if row["model_name"] == INCUMBENT else ""
        short = row["model_name"].split("/")[-1]
        lines.append(
            f"| `{short}`{mark} | {row['embedding_dim']} | {row['max_seq_length']} | "
            f"{row['chunk_token_budget']} | {row['n_chunks']} | "
            f"{row['tier_a']['hits@5']}/{n_a} ({row['tier_a']['recall@5']:.2f}) | "
            f"**{row['tier_b']['hits@5']}/{n_b}** ({row['tier_b']['recall@5']:.2f}) | "
            f"{row['tier_b']['mrr@10']:.3f} |"
        )
    return "\n".join(lines)


def cost_table(results: list[dict[str, Any]]) -> str:
    """The axes that decide whether a quality win is affordable."""
    lines = [
        "| Model | Weights (MB) | Ingest (s) | Peak RSS (MB) | Index (KB) | "
        "Embed p50 (ms) | Search p50 (ms) | Ratio |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in results:
        latency = row.get("latency_ms", {})
        lines.append(
            f"| `{row['model_name'].split('/')[-1]}` | {_mb(row['model_disk_bytes'])} | "
            f"{row['ingestion_seconds']} | {_mb(row['peak_rss_bytes'])} | "
            f"{row['index_bytes'] / 1024:.0f} | {latency.get('embed_p50', '-')} | "
            f"{latency.get('search_p50', '-')} | {latency.get('embed_over_search', '-')}x |"
        )
    return "\n".join(lines)


def apply_decision_rule(results: list[dict[str, Any]]) -> tuple[str, list[str]]:
    """Return the verdict and the line of reasoning that produced it."""
    incumbent = next((r for r in results if r["model_name"] == INCUMBENT), None)
    if incumbent is None:
        return "INCONCLUSIVE", ["the incumbent was not evaluated, so there is no baseline"]

    base = incumbent["tier_b"][PRIMARY_METRIC]
    reasoning: list[str] = [
        f"incumbent Tier B {PRIMARY_METRIC}: {base}/{incumbent['tier_b']['n']}",
        f"a candidate must exceed it by more than {MARGIN_QUERIES} queries to switch",
    ]

    qualified = []
    for row in results:
        if row["model_name"] == INCUMBENT:
            continue
        margin = row["tier_b"][PRIMARY_METRIC] - base
        blockers = _cost_blockers(row, incumbent)
        verdict = "clears" if margin > MARGIN_QUERIES and not blockers else "rejected"
        detail = ", ".join(blockers) if blockers else f"margin {margin:+d}"
        reasoning.append(f"{row['model_name'].split('/')[-1]}: {verdict} ({detail})")
        if verdict == "clears":
            qualified.append(row)

    if not qualified:
        return "KEEP THE INCUMBENT", reasoning
    winner = min(qualified, key=lambda r: (-r["tier_b"][PRIMARY_METRIC], r["model_disk_bytes"] or 0))
    return f"SWITCH TO {winner['model_name']}", reasoning


def _cost_blockers(row: dict[str, Any], incumbent: dict[str, Any]) -> list[str]:
    """Which pre-registered cost gates this candidate fails, if any."""
    blockers: list[str] = []
    if row["ingestion_seconds"] > incumbent["ingestion_seconds"] * MAX_INGEST_RATIO:
        blockers.append(f"ingestion {row['ingestion_seconds']}s exceeds 2x incumbent")
    if row.get("model_disk_bytes") and row["model_disk_bytes"] > MAX_MODEL_BYTES:
        blockers.append(f"weights {_mb(row['model_disk_bytes'])} MB exceed the 1.5 GB ceiling")
    rss, base_rss = row.get("peak_rss_bytes"), incumbent.get("peak_rss_bytes")
    if rss and base_rss and rss > base_rss * MAX_RSS_RATIO:
        blockers.append(f"peak RSS {_mb(rss)} MB exceeds 2x incumbent")
    return blockers


def main(argv: list[str] | None = None) -> int:
    """Print the two tables and the mechanically-applied verdict."""
    parser = argparse.ArgumentParser(description="Compare evaluated models.")
    parser.add_argument("--results-dir", type=Path, default=Path("eval/results"))
    args = parser.parse_args(argv)

    results = load_results(args.results_dir)
    if not results:
        print(f"no results in {args.results_dir}")
        return 1

    print("## Retrieval quality\n")
    print(quality_table(results))
    print("\n## Cost\n")
    print(cost_table(results))

    verdict, reasoning = apply_decision_rule(results)
    print("\n## Decision\n")
    for line in reasoning:
        print(f"- {line}")
    print(f"\n**{verdict}**")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
