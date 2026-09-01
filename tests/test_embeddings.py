"""The embedding seam: budgets derived from the model, and asymmetric prefixes.

These tests run without downloading anything. What they pin is the arithmetic
and the string handling that decide whether a query and a passage land in the
same vector space.
"""

from __future__ import annotations

import pytest

from pdf_search.chunking import CHUNK_TOKEN_BUDGET
from pdf_search.embeddings import (
    BASELINE_MODEL_NAME,
    DEFAULT_MODEL_NAME,
    PrefixScheme,
    budget_for,
    chunk_budget_for,
    prefixes_for,
)


def test_the_derived_budget_reproduces_the_incumbent_constant() -> None:
    """128 less two special tokens less the headroom is exactly today's budget.

    The budget stops being a hand-picked constant and becomes a function of the
    model, without changing what the shipped model gets.
    """
    assert budget_for(128) == CHUNK_TOKEN_BUDGET == 110


def test_a_wider_window_yields_a_wider_budget() -> None:
    """A 512-token model earns 494 tokens of content on the same rule."""
    assert budget_for(512) == 494


def test_a_window_too_small_to_carry_content_is_refused() -> None:
    """A budget of zero or less is a configuration error, not a small chunk."""
    with pytest.raises(ValueError):
        budget_for(8)


def test_the_baseline_model_uses_no_prefixes() -> None:
    """The model the brief suggests has no notion of a query role at all."""
    assert prefixes_for(BASELINE_MODEL_NAME) == PrefixScheme(query="", document="")


def test_the_default_model_is_asymmetric() -> None:
    """The shipped model is retrieval-trained, so its two sides differ."""
    scheme = prefixes_for(DEFAULT_MODEL_NAME)
    assert scheme.query and scheme.document
    assert scheme.query != scheme.document


def test_the_shipped_budget_is_measured_not_derived() -> None:
    """The window would allow 494; the corpus measured better at 110.

    Regression guard for a tempting simplification: deriving the budget from the
    window alone would silently triple the chunk size and, on this corpus, cost
    five of twenty-six paraphrase queries.
    """
    assert budget_for(512) == 494
    assert chunk_budget_for(DEFAULT_MODEL_NAME, 512) == 110


def test_a_model_without_a_measured_budget_falls_back_to_the_window() -> None:
    """Nothing is pinned for a model this corpus was never measured against."""
    assert chunk_budget_for("some/unlisted-model", 512) == budget_for(512)


def test_a_measured_budget_never_exceeds_what_the_window_allows() -> None:
    """A measurement cannot license truncation if the window is later narrower."""
    assert chunk_budget_for(DEFAULT_MODEL_NAME, 64) == budget_for(64)


def test_e5_models_distinguish_queries_from_passages() -> None:
    """Omitting these is silent: no error, just quietly worse retrieval."""
    scheme = prefixes_for("intfloat/multilingual-e5-small")
    assert scheme.query == "query: "
    assert scheme.document == "passage: "


def test_solon_prefixes_the_query_only() -> None:
    """Solon documents a space before the colon, and no passage prefix."""
    scheme = prefixes_for("OrdalieTech/Solon-embeddings-base-0.1")
    assert scheme.query == "query : "
    assert scheme.document == ""


def test_an_unrecognised_model_defaults_to_no_prefixes() -> None:
    """Unknown models get the conservative scheme, recorded in the manifest."""
    assert prefixes_for("some/unlisted-model") == PrefixScheme()


def test_an_empty_scheme_leaves_text_untouched() -> None:
    scheme = PrefixScheme()
    assert scheme.for_documents(["un", "deux"]) == ["un", "deux"]
    assert scheme.for_query("un") == "un"


def test_a_scheme_applies_its_two_sides_independently() -> None:
    scheme = PrefixScheme(query="query: ", document="passage: ")
    assert scheme.for_documents(["texte"]) == ["passage: texte"]
    assert scheme.for_query("texte") == "query: texte"
