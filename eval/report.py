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

# What src/pdf_search/embeddings.py actually defaults to.
SHIPPED_MODEL = "intfloat/multilingual-e5-small"
MAX_INGEST_RATIO = 2.0
MAX_RSS_RATIO = 2.0
MAX_MODEL_BYTES = 1.5 * 1024**3

BYTES_PER_MB = 1024 * 1024


def load_results(directory: Path, ablation: bool = False) -> list[dict[str, Any]]:
    """Read results, incumbent first.

    Runs with a forced budget are the ablation, not the sweep. They answer a
    different question -- what a model does at a fixed chunk size -- and folding
    them into the decision table would enter the same model twice under two
    configurations.
    """
    results = []
    for path in sorted(directory.glob("*.json")):
        if "--smoke" in path.name:
            continue
        if ("--budget" in path.name) != ablation:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        # The same directory also holds the OCR bake-off, which scores engines
        # rather than models and carries none of the keys below. Filtering on
        # the payload's shape rather than on its filename keeps this correct
        # the next time a differently-named artifact lands here.
        if "model_name" not in payload:
            continue
        results.append(payload)
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


def _print_divergence(
    verdict: str, results: list[dict[str, Any]], ablation: list[dict[str, Any]]
) -> None:
    """Say so when the shipped model is not the one this rule returns.

    The rule runs as pre-registered and its verdict is never adjusted to match
    what shipped. But the rule scores each model at its own window-derived
    budget, and the shipped model runs at a measured 110, so the row it judges
    is not the row that ships. Rather than ask the reader to take the override
    on trust, the same gates are applied to the shipped configuration here.
    """
    if SHIPPED_MODEL in verdict:
        return
    print("\n## The rule against the shipped configuration\n")
    print(
        f"The model that ships is `{SHIPPED_MODEL}` at a measured budget of 110, not the\n"
        "window-derived 494 the sweep above scores it at. Applying the same gates to that row:\n"
    )

    incumbent = next((r for r in results if r["model_name"] == INCUMBENT), None)
    shipped = next((r for r in ablation if r["model_name"] == SHIPPED_MODEL), None)
    if incumbent is None or shipped is None:
        print("- the shipped configuration was not evaluated, so the gates cannot be applied")
        return

    blockers = _cost_blockers(shipped, incumbent)
    margin = shipped["tier_b"][PRIMARY_METRIC] - incumbent["tier_b"][PRIMARY_METRIC]
    margin_passes = margin > MARGIN_QUERIES
    print(f"- cost gates: {', '.join(blockers) if blockers else 'all three pass'}")
    print(
        f"- margin gate: {margin:+d}, and more than {MARGIN_QUERIES} is required -- "
        f"{'passes' if margin_passes else 'fails'}"
    )
    if blockers or margin_passes:
        return
    print(
        "\nSo the shipped configuration is not blocked by the two gates that later proved\n"
        "miscalibrated; it is blocked only by the margin gate, which was calibrated correctly.\n"
        "The switch is therefore one explicit judgement and not a rescued gate: a Tier B margin\n"
        f"of {margin:+d} is inside the noise of n=26, and the case rests on Tier A instead. That\n"
        "argument, and the miscalibration of the other two gates, is in eval/DECISION.md and in\n"
        "the README under 'Why this model, and how I know'."
    )


def main(argv: list[str] | None = None) -> int:
    """Print the two tables and the mechanically-applied verdict."""
    parser = argparse.ArgumentParser(description="Compare evaluated models.")
    parser.add_argument("--results-dir", type=Path, default=Path("eval/results"))
    args = parser.parse_args(argv)

    results = load_results(args.results_dir)
    if not results:
        print(f"no results in {args.results_dir}")
        return 1

    print("## Retrieval quality, each model at the budget its own window earns\n")
    print(quality_table(results))
    print("\n## Cost\n")
    print(cost_table(results))

    ablation = load_results(args.results_dir, ablation=True)
    if ablation:
        incumbent = [r for r in results if r["model_name"] == INCUMBENT]
        print("\n## Ablation: the same models held at the incumbent chunk budget\n")
        print("A candidate does not only embed differently, it also re-chunks the")
        print("corpus, so model and chunk size are confounded in the table above.")
        print("Holding the budget at 110 leaves the model as the only variable.\n")
        print(quality_table(incumbent + ablation))

    verdict, reasoning = apply_decision_rule(results)
    print("\n## Decision\n")
    for line in reasoning:
        print(f"- {line}")
    print(f"\n**{verdict}**")
    _print_divergence(verdict, results, ablation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
