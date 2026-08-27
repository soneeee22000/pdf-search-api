"""Text hygiene that measurably affects embedding quality for French documents."""

from __future__ import annotations

from pdf_search.pdf_text import normalise


def test_nfkc_folds_the_no_break_spaces_french_typography_requires() -> None:
    """U+00A0 and U+202F appear before ':' and '?' in every French document."""
    result = normalise("Article : le conseil approuve")

    assert " " not in result
    assert " " not in result
    assert result == "Article : le conseil approuve"


def test_nfkc_resolves_ligatures() -> None:
    """The fi ligature is a compatibility character, so NFKC decomposes it."""
    assert normalise("conﬁrmation") == "confirmation"


def test_soft_hyphens_are_removed() -> None:
    """U+00AD survives NFKC and corrupts tokenisation from inside words."""
    assert normalise("adminis­tration") == "administration"


def test_line_wrap_hyphenation_is_joined() -> None:
    """A word broken across a line break must be rejoined before embedding."""
    assert normalise("adminis-\ntration communale") == "administration communale"


def test_carriage_returns_are_normalised_before_the_hyphen_rules_run() -> None:
    """PDF extractors emit CRLF, and the hyphen rule anchors on a bare newline.

    Regression: de-hyphenation silently did nothing on real PDFs, because the
    synthetic test input used '\\n' and the extractor emits '\\r\\n'. A test
    against a real PDF fixture is what surfaced it.
    """
    assert normalise("adminis-\r\ntration communale") == "administration communale"


def test_no_carriage_returns_survive_normalisation() -> None:
    """Stray carriage returns must not reach the index or the API response."""
    assert "\r" not in normalise("premiere ligne\r\nseconde ligne")


def test_genuine_compound_hyphens_survive() -> None:
    """'sous-prefet' keeps its hyphen; only lowercase line wraps are joined."""
    result = normalise("le sous-prefet a signe")

    assert "sous-prefet" in result


def test_compound_across_a_line_break_is_a_known_false_positive() -> None:
    """Documented limitation: a genuine compound wrapped at a hyphen is merged.

    Distinguishing 'sous-\\nprefet' from 'adminis-\\ntration' needs a lexicon.
    The rule is deliberately simple and this is the cost.
    """
    assert normalise("le sous-\nprefet a signe") == "le sousprefet a signe"


def test_typographic_apostrophes_are_unified() -> None:
    """U+2019 is not touched by NFKC and must be normalised deliberately."""
    assert normalise("l’article") == "l'article"


def test_whitespace_runs_collapse() -> None:
    """Extraction artefacts leave runs of spaces that add nothing."""
    assert normalise("le    conseil     municipal") == "le conseil municipal"
