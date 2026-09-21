"""End-to-end pipeline tests with a fake ModelClient.

These exercise the real agent -- guards, execution, the repair loop, every
error path -- against the real database, with only the model faked. No API key
required, so the wiring is covered in CI.
"""

from __future__ import annotations

import pytest

from bse_nlq.agent import NLQAgent, Outcome
from bse_nlq.config import SETTINGS
from bse_nlq.errors import ModelError
from tests.fakes import FakeModelClient, declined, plan


def make_agent(db, *payloads) -> NLQAgent:
    return NLQAgent(db, SETTINGS, client=FakeModelClient(*payloads))


# --- happy path -------------------------------------------------------------

def test_happy_path_returns_rows_and_answer(db):
    agent = make_agent(db, plan("SELECT COUNT(*) AS n FROM venues"), "There are 2 venues.")
    result = agent.ask("How many venues are there?")

    assert result.outcome is Outcome.ANSWERED
    assert result.ok and result.answerable
    assert result.rows == ((2,),)
    assert result.columns == ("n",)
    assert result.answer == "There are 2 venues."
    assert result.assumptions == ("Counts every venue on file.",)


def test_usage_accumulates_across_both_calls(db):
    agent = make_agent(db, plan("SELECT 1 AS n"), "ok")
    usage = agent.ask("anything").usage
    assert usage.calls == 2                 # plan + synthesize
    assert usage.input_tokens == 200
    assert usage.cache_read_tokens == 20
    assert usage.cache_hit


def test_limit_is_injected_into_generated_sql(db):
    agent = make_agent(db, plan("SELECT event_id FROM events"), "ok")
    result = agent.ask("List events")
    assert "LIMIT" in result.sql.upper()
    assert result.row_count <= SETTINGS.max_rows


def test_schema_prompt_is_stable_and_date_is_in_the_user_turn(db):
    agent = make_agent(db, plan("SELECT 1 AS n"), "ok")
    agent.ask("anything")
    first = agent.client.requests[0]
    assert "Today's date is" in first.messages[0].content
    assert "Today's date is" not in first.system
    assert first.response_schema is not None       # structured outputs on planning
    assert agent.client.requests[1].response_schema is None   # free text on synthesis


# --- declining --------------------------------------------------------------

def test_unanswerable_question_is_declined_without_touching_the_database(db):
    agent = make_agent(db, declined("No marketing data exists."))
    result = agent.ask("Which campaign drove sales?")

    assert result.outcome is Outcome.DECLINED
    assert result.ok is True          # declining cleanly is correct behaviour
    assert result.answerable is False
    assert result.sql == ""
    assert "No marketing data exists." in result.answer
    assert "ticketing" in result.answer               # scope description appended
    assert len(agent.client.requests) == 1            # no synthesis call


# --- error paths ------------------------------------------------------------

def test_empty_result_set_is_flagged_to_the_synthesis_step(db):
    agent = make_agent(
        db,
        plan("SELECT * FROM events WHERE event_date = '1900-01-01'"),
        "No matching events were found.",
    )
    result = agent.ask("Events in 1900?")

    assert result.outcome is Outcome.ANSWERED
    assert result.is_empty
    # The synthesis prompt must say 0 rows explicitly so the model cannot
    # report an invented zero as a business fact.
    assert "returned 0 rows" in agent.client.requests[1].messages[0].content


def test_unsafe_sql_is_rejected_before_execution(db):
    agent = make_agent(db, plan("DROP TABLE events"))
    result = agent.ask("Delete everything")

    assert result.outcome is Outcome.FAILED
    assert "read-only" in result.answer.lower()
    assert len(agent.client.requests) == 1   # never reached synthesis


def test_repair_loop_recovers_from_a_bad_column(db):
    agent = make_agent(
        db,
        plan("SELECT no_such_column FROM events"),   # fails
        plan("SELECT COUNT(*) AS n FROM events"),    # repaired
        "There are events on file.",
    )
    result = agent.ask("How many events?")

    assert result.outcome is Outcome.ANSWERED
    assert result.rows[0][0] > 0
    assert result.repaired
    assert len(result.attempts) == 2
    assert "no_such_column" in result.attempts[0].error
    repair_turn = agent.client.requests[1].messages[-1].content
    assert "SQLite error" in repair_turn and "no_such_column" in repair_turn


def test_repair_loop_gives_up_after_the_configured_limit(db):
    agent = make_agent(db, plan("SELECT bad_one FROM events"), plan("SELECT bad_two FROM events"))
    result = agent.ask("How many events?")

    assert result.outcome is Outcome.FAILED
    assert len(result.attempts) == 2                 # max_repair_attempts = 1
    assert len(agent.client.requests) == 2           # no synthesis call
    assert "bad_two" in result.answer
    assert result.sql                                # failing SQL still surfaced


def test_declining_during_repair_is_reported(db):
    agent = make_agent(db, plan("SELECT nope FROM events"), declined("Cannot be fixed."))
    result = agent.ask("How many events?")
    assert result.outcome is Outcome.FAILED
    assert "Cannot be fixed." in result.answer


def test_blank_question_is_handled_without_calling_the_model(db):
    agent = make_agent(db)
    result = agent.ask("   ")
    assert result.outcome is Outcome.FAILED
    assert "enter a question" in result.answer.lower()
    assert agent.client.requests == []


def test_malformed_model_output_is_a_clean_error(db):
    agent = make_agent(db, "this is not json")
    result = agent.ask("anything")
    assert result.outcome is Outcome.FAILED
    assert "could not read" in result.answer.lower()


def test_model_errors_surface_as_readable_text(db):
    agent = make_agent(db, ModelError("Rate limited by the Claude API."))
    result = agent.ask("anything")
    assert result.outcome is Outcome.FAILED
    assert result.answer == "Rate limited by the Claude API."


@pytest.mark.parametrize("truncating_sql", ["SELECT event_id FROM events"])
def test_row_cap_is_disclosed_as_an_assumption(db, truncating_sql):
    agent = make_agent(db, plan(truncating_sql), "Here are the events.")
    result = agent.ask("List every event")
    assert result.truncated
    assert any("capped at" in a for a in result.assumptions)
