"""The README stakes its evaluation on `eval/report.py` being runnable.

It imports nothing heavier than json, so pinning its behaviour costs the suite
nothing and keeps the offer in the README honest: a reviewer who runs it gets
the table, not a traceback.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from eval.report import INCUMBENT, load_results

RESULTS = Path(__file__).resolve().parents[1] / "eval" / "results"


def test_the_committed_results_load_without_error() -> None:
    """The script the README tells a reviewer to run must run on this repo.

    It broke once: `load_results` globbed every JSON in the directory, so the
    OCR bake-off results -- which score engines and carry no `model_name` --
    were fed into a sort keyed on that field. The report died with a KeyError
    before printing anything, while the README pointed at it as the proof the
    decision rule was applied mechanically.
    """
    sweep = load_results(RESULTS)

    assert sweep, "no model results loaded"
    assert all("model_name" in row for row in sweep)
    assert sweep[0]["model_name"] == INCUMBENT, "the incumbent must sort first"


def test_the_ocr_results_are_not_mistaken_for_model_results() -> None:
    """OCR artifacts share the directory and must be ignored by both views."""
    ocr_files = sorted(RESULTS.glob("ocr--*.json"))
    assert ocr_files, "the OCR results are missing, so this guards nothing"

    engines = {json.loads(p.read_text(encoding="utf-8"))["engine"] for p in ocr_files}
    loaded = {row["model_name"] for row in load_results(RESULTS)}
    loaded |= {row["model_name"] for row in load_results(RESULTS, ablation=True)}

    assert not (engines & loaded), "an OCR engine leaked into the model comparison"


def test_the_ablation_and_the_sweep_are_disjoint_views() -> None:
    """A forced-budget run answers a different question and must not double-count."""
    sweep = load_results(RESULTS)
    ablation = load_results(RESULTS, ablation=True)

    assert ablation, "no ablation results loaded"
    assert all(row["chunk_token_budget"] == 110 for row in ablation)
    incumbent_rows = [r for r in sweep if r["model_name"] == INCUMBENT]
    assert len(incumbent_rows) == 1, "the incumbent must appear exactly once in the sweep"


@pytest.mark.parametrize("ablation", [False, True])
def test_loading_an_empty_directory_is_not_an_error(tmp_path: Path, ablation: bool) -> None:
    """A missing result set is reported by the caller, not raised here."""
    assert load_results(tmp_path, ablation=ablation) == []
