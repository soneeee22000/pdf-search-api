"""The README makes checkable claims. This checks the one that kept drifting."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"

# A plain `pytest` run collects the whole suite; anything that narrows the
# selection makes the collected count a subset, and comparing that to the
# README would fail for the wrong reason.
_WHOLE_SUITE_ARGS = {"", ".", "tests"}


def _is_whole_suite(config: pytest.Config) -> bool:
    """Whether this run collected every test, rather than a filtered subset."""
    if config.option.keyword or config.option.markexpr:
        return False
    if getattr(config.option, "lf", False) or getattr(config.option, "ff", False):
        return False
    return all(arg in _WHOLE_SUITE_ARGS for arg in config.args)


def test_the_readme_states_the_number_of_tests_that_exist(
    request: pytest.FixtureRequest,
) -> None:
    """The README's test count is enforced here instead of maintained by hand.

    It drifted three times as this suite grew -- 78, 81, 83 -- each time because
    a test was added and the prose was not. A README that is provably wrong
    about the one number a reviewer can check in a single command devalues every
    number they cannot check as easily, so the claim is pinned to reality rather
    than to my memory.
    """
    if not _is_whole_suite(request.config):
        pytest.skip("run selection is filtered, so the collected count is not the suite total")

    collected = request.session.testscollected
    stated = {int(match) for match in re.findall(r"\b(\d+) tests\b", README.read_text(encoding="utf-8"))}

    assert stated, "the README no longer states a test count"
    assert stated == {collected}, (
        f"README states {sorted(stated)} tests; the suite collects {collected}. "
        "Update the README, or this claim is false the moment a reviewer runs pytest."
    )
