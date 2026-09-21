"""Rendering must not lose or mangle model-generated text.

Answers and SQLite error messages are arbitrary text as far as rich is
concerned. Square brackets in them were previously parsed as markup tags and
silently dropped -- these tests pin the escaping fix.
"""

from __future__ import annotations

import pytest
from rich.console import Console

from bse_nlq import cli
from bse_nlq.agent import Attempt, NLQResult, Outcome


@pytest.fixture
def capture(monkeypatch):
    console = Console(record=True, width=120, force_terminal=False, no_color=True)
    monkeypatch.setattr(cli, "console", console)
    return console


def answered(**kwargs) -> NLQResult:
    return NLQResult(question="q", outcome=Outcome.ANSWERED, **kwargs)


def test_brackets_in_an_answer_are_not_swallowed(capture):
    cli.render(answered(answer="Revenue was $1.2M [see note]; the tag [red] is literal."))
    out = capture.export_text()
    assert "[see note]" in out
    assert "[red]" in out


def test_brackets_in_a_sqlite_error_survive(capture):
    cli.render(answered(
        answer="Recovered.",
        sql="SELECT 1",
        attempts=(Attempt(sql="SELECT x", error="no such column: [x]"), Attempt(sql="SELECT 1")),
    ))
    assert "[x]" in capture.export_text()


def test_brackets_in_assumptions_survive(capture):
    cli.render(answered(answer="ok", sql="SELECT 1",
                        assumptions=("Excludes comps [is_comp = 1].",)))
    assert "[is_comp = 1]" in capture.export_text()


def test_unclosed_markup_does_not_raise(capture):
    # An unescaped "[/bold" would make rich raise MarkupError mid-render.
    cli.render(answered(answer="Total [/bold unclosed and [unmatched"))
    assert "unclosed" in capture.export_text()


def test_answer_and_sql_are_both_shown(capture):
    cli.render(answered(
        answer="There are 2 venues.",
        sql="SELECT COUNT(*) FROM venues",
        columns=("n",), rows=((2,),),
    ))
    out = capture.export_text()
    assert "There are 2 venues." in out
    assert "SELECT" in out and "venues" in out   # transparency: SQL is always shown


def test_repair_notice_is_only_shown_when_a_repair_happened(capture):
    cli.render(answered(answer="ok", sql="SELECT 1", attempts=(Attempt(sql="SELECT 1"),)))
    assert "retried" not in capture.export_text()


def test_show_sql_only_prints_just_the_query(capture):
    cli.render(answered(answer="There are 2 venues.", sql="SELECT 1"), show_sql_only=True)
    out = capture.export_text()
    assert "SELECT 1" in out
    assert "There are 2 venues." not in out


@pytest.mark.parametrize("outcome", list(Outcome))
def test_every_outcome_has_a_panel_style(outcome):
    assert outcome in cli._OUTCOME_STYLE
