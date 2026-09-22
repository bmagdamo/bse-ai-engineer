"""Prompt context must reflect the live schema, including the value hints."""

from __future__ import annotations

from bse_nlq.prompts import build_question_turn, build_system_prompt
from bse_nlq.schema_context import build_schema_context, render_value_hints


def test_context_lists_every_table(db):
    context = build_schema_context(db)
    for table in ("venues", "teams", "events", "orders", "tickets",
                  "customers", "seating_sections", "event_categories"):
        assert f"TABLE {table} " in context


def test_context_includes_foreign_keys(db):
    assert "FOREIGN KEY home_team_id -> teams.team_id" in build_schema_context(db)


def test_value_hints_enumerate_real_literals(db):
    hints = render_value_hints(db)
    # This is the guard against the model inventing status = 'complete'.
    assert "orders.status: 'cancelled', 'completed', 'refunded'" in hints
    assert "'Brooklyn Nets'" in hints
    assert "'Barclays Center'" in hints
    assert "'box_office'" in hints


def test_context_states_the_data_coverage_window(db):
    assert "events.event_date spans" in build_schema_context(db)


def test_system_prompt_carries_the_semantic_layer(db):
    prompt = build_system_prompt(build_schema_context(db))
    assert "Metric definitions" in prompt
    assert "is_comp = 0" in prompt
    assert "SQLite date recipes" in prompt
    assert "start of month" in prompt
    # Few-shots must cover the declining path, not just happy paths.
    assert '"is_answerable": false' in prompt


def test_system_prompt_is_byte_stable(db):
    """The cache breakpoint sits on the system prompt, so two builds against
    the same database must be byte-identical -- any per-request value leaking
    in here would invalidate the prompt cache on every question."""
    assert build_system_prompt(build_schema_context(db)) == build_system_prompt(
        build_schema_context(db)
    )


def test_todays_date_lives_in_the_user_turn():
    """The one volatile value goes after the cached prefix, not inside it."""
    turn = build_question_turn(
        "How many tickets sold last month?", "2026-09-18", "America/New_York")
    assert "2026-09-18" in turn
    assert "America/New_York" in turn
    assert "How many tickets sold last month?" in turn



def test_prompt_forbids_the_utc_now_helper():
    """The regression this guards: the prompt used to hand the model
    date('now'), which SQLite evaluates in UTC, while telling it today's date
    in local time. The two disagree for part of every day, and on a month
    boundary that silently shifts a "last month" window by a whole month.

    The date recipes must therefore anchor on the literal from the user turn.
    """
    from bse_nlq.prompts import DATE_RECIPES, FEW_SHOT_EXAMPLES

    assert "date('now'" not in DATE_RECIPES.replace("NEVER write date('now')", "")
    assert "date('now'" not in FEW_SHOT_EXAMPLES
    assert "TODAY" in DATE_RECIPES
