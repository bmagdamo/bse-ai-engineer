"""End-to-end pipeline tests with a fake ModelClient.

These exercise the real agent -- guards, execution, the repair loop, every
error path -- against the real database, with only the model faked. No API key
required, so the wiring is covered in CI.
"""

from __future__ import annotations

import pytest

from bse_nlq.agent import NLQAgent, Outcome
from bse_nlq.config import SETTINGS, Settings
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


# --- row cap disclosure -----------------------------------------------------
# These run with db.max_rows == settings.max_rows, which is how cli.py and
# app.py actually wire it. The older test above passes only because its
# fixture gives the Database a *smaller* cap (50) than settings (200), which
# is what let the production bug hide: guards.enforce_limit puts
# LIMIT <max_rows> into the SQL, so Database can never fetch the extra row it
# uses to detect truncation, and result.truncated was always False.

@pytest.fixture
def capped(db_path):
    """A Database and Settings that agree on the cap, as production does."""
    from bse_nlq.db import Database

    settings = Settings(max_rows=5, db_path=SETTINGS.db_path)
    with Database(db_path, max_rows=settings.max_rows, timeout_seconds=10) as handle:
        yield handle, settings


def test_row_cap_is_disclosed_when_db_and_settings_agree(capped):
    db, settings = capped
    total = db.run_select("SELECT COUNT(*) FROM events").rows[0][0]
    assert total > settings.max_rows, "fixture needs a table bigger than the cap"

    agent = NLQAgent(db, settings,
                     client=FakeModelClient(plan("SELECT event_id FROM events"), "ok"))
    result = agent.ask("List every event")

    assert result.row_count == settings.max_rows
    assert result.truncated, "the user was never told the list was partial"
    assert any("capped at" in a for a in result.assumptions)


def test_a_model_authored_top_n_is_not_reported_as_truncated(capped):
    """A "top 3" question returns three rows because three were asked for.
    Calling that truncated would tell the user their complete answer is
    partial."""
    db, settings = capped
    agent = NLQAgent(db, settings,
                     client=FakeModelClient(plan("SELECT event_id FROM events LIMIT 3"), "ok"))
    result = agent.ask("Top 3 events")

    assert result.row_count == 3
    assert not result.truncated
    assert not any("capped at" in a for a in result.assumptions)


# --- settings propagation ---------------------------------------------------

def test_agent_settings_reach_the_model_client(db, monkeypatch):
    """ClaudeClient used to default to the import-time SETTINGS singleton, so
    an agent built with its own Settings got someone else's retry policy and
    request timeout."""
    import anthropic

    from bse_nlq.claude import ClaudeClient

    monkeypatch.setattr(anthropic, "Anthropic", lambda **_kwargs: object())
    settings = Settings(max_attempts=7, request_timeout_seconds=3.0,
                        model="claude-haiku-4-5-20251001")
    agent = NLQAgent(db, settings)

    assert isinstance(agent.client, ClaudeClient)
    assert agent.client.settings is settings
    assert agent.client.settings.max_attempts == 7


# --- the business calendar --------------------------------------------------

def test_todays_date_is_resolved_in_the_business_timezone(db):
    """The prompt's "today" and the SQL's "today" must be the same day. The
    agent resolves it once, in the configured zone, and hands the model a
    literal -- date('now') (UTC) is banned by the prompt."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    settings = Settings(timezone="Pacific/Kiritimati")   # UTC+14, far from UTC
    agent = NLQAgent(db, settings, client=FakeModelClient(plan("SELECT 1 AS n"), "ok"))
    agent.ask("anything")

    expected = datetime.now(ZoneInfo("Pacific/Kiritimati")).date().isoformat()
    turn = agent.client.requests[0].messages[0].content
    assert expected in turn
    assert "Pacific/Kiritimati" in turn


# --- provenance -------------------------------------------------------------

def test_every_answer_carries_the_model_and_prompt_that_produced_it(db):
    agent = make_agent(db, plan("SELECT 1 AS n"), "ok")
    result = agent.ask("anything")
    assert result.model == SETTINGS.model
    assert result.prompt_version == agent.prompt_version
    assert len(result.request_id) == 12
    assert result.total_seconds > 0


def test_the_answering_model_is_recorded_not_the_configured_one(db):
    """When the fallback serves a question, an audit line naming the primary
    attributes the answer to a model that never saw it."""
    agent = NLQAgent(db, SETTINGS, client=FakeModelClient(
        plan("SELECT 1 AS n"), "ok", model="claude-haiku-4-5"))
    result = agent.ask("anything")
    assert result.model == "claude-haiku-4-5" != SETTINGS.model


def test_a_question_that_never_reached_a_model_falls_back_to_the_configured_one(db):
    assert make_agent(db).ask("   ").model == SETTINGS.model


def test_each_question_gets_its_own_correlation_id(db):
    agent = make_agent(db, plan("SELECT 1 AS n"), "ok", plan("SELECT 2 AS n"), "ok")
    first = agent.ask("one")
    second = agent.ask("two")
    assert first.request_id != second.request_id


def test_an_empty_question_is_still_traceable(db):
    assert make_agent(db).ask("   ").request_id


# --- the repeat-question cache ---------------------------------------------

def test_an_identical_question_is_served_without_calling_the_model(db):
    agent = make_agent(db, plan("SELECT COUNT(*) AS n FROM venues"), "There are 2 venues.")
    first = agent.ask("How many venues are there?")
    second = agent.ask("How many venues are there?")   # no payloads left to serve

    assert second.answer == first.answer
    assert second.rows == first.rows
    assert second.cached and not first.cached
    assert len(agent.client.requests) == 2, "the model was not called a second time"


def test_the_cache_is_case_insensitive_but_not_question_blind(db):
    agent = make_agent(db, plan("SELECT 1 AS n"), "ok", plan("SELECT 2 AS n"), "ok")
    agent.ask("How many venues?")
    assert agent.ask("HOW MANY VENUES?").cached
    assert not agent.ask("How many events?").cached, "a different question is a miss"


def test_a_cached_answer_still_gets_a_fresh_request_id(db):
    agent = make_agent(db, plan("SELECT 1 AS n"), "ok")
    first = agent.ask("q")
    second = agent.ask("q")
    assert second.cached
    assert second.request_id != first.request_id, "each request is its own audit event"


def test_failures_are_never_cached(db):
    """Replaying a failure would hide a transient outage behind a stale error
    for the whole TTL."""
    agent = make_agent(db, ModelError("API is down"),
                       plan("SELECT 1 AS n"), "recovered")
    assert agent.ask("q").outcome is Outcome.FAILED
    assert agent.ask("q").answer == "recovered"


def test_caching_can_be_switched_off(db):
    settings = Settings(cache_ttl_seconds=0)
    agent = NLQAgent(db, settings, client=FakeModelClient(
        plan("SELECT 1 AS n"), "ok", plan("SELECT 1 AS n"), "ok"))
    agent.ask("q")
    assert not agent.ask("q").cached
    assert len(agent.client.requests) == 4


# --- schema drift -----------------------------------------------------------

def test_a_changed_schema_rebuilds_the_prompt_and_drops_cached_answers(db, monkeypatch):
    """Streamlit caches one agent for the life of the process, so a migration
    must not leave it answering from a schema it no longer describes."""
    settings = Settings(schema_check_interval_seconds=0.001)
    agent = NLQAgent(db, settings, client=FakeModelClient(
        plan("SELECT 1 AS n"), "ok", plan("SELECT 1 AS n"), "ok"))
    agent.ask("q")
    assert len(agent.cache) == 1
    before = agent.schema_fingerprint

    monkeypatch.setattr("bse_nlq.schema_context.fingerprint", lambda _db: "migrated")
    assert not agent.ask("q").cached, "the pre-migration answer must not be served"
    assert agent.schema_fingerprint == "migrated" != before


def test_the_schema_check_can_be_switched_off(db, monkeypatch):
    agent = NLQAgent(db, Settings(schema_check_interval_seconds=0),
                     client=FakeModelClient(plan("SELECT 1 AS n"), "ok"))
    before = agent.schema_fingerprint
    monkeypatch.setattr("bse_nlq.schema_context.fingerprint", lambda _db: "migrated")
    agent.ask("q")
    assert agent.schema_fingerprint == before


# --- the end-to-end deadline ------------------------------------------------

def test_an_exhausted_budget_fails_cleanly_before_calling_the_model(db, monkeypatch):
    from bse_nlq.deadline import Deadline

    monkeypatch.setattr(Deadline, "after", classmethod(lambda cls, _s: cls(0.0)))
    agent = make_agent(db, plan("SELECT 1 AS n"), "ok")
    result = agent.ask("q")

    assert result.outcome is Outcome.FAILED
    assert "too long" in result.answer
    assert agent.client.requests == [], "no budget left means no call is started"


def test_the_deadline_travels_with_the_request(db):
    agent = make_agent(db, plan("SELECT 1 AS n"), "ok")
    agent.ask("q")
    assert all(r.deadline is not None for r in agent.client.requests), (
        "both the planning and the synthesis call share one budget"
    )


# --- audit ------------------------------------------------------------------

def test_every_question_is_audited(db, tmp_path):
    import json

    from bse_nlq.observability import AuditLog

    path = tmp_path / "audit.jsonl"
    agent = NLQAgent(db, SETTINGS, client=FakeModelClient(
        plan("SELECT COUNT(*) AS n FROM venues"), "There are 2 venues.",
        declined("No marketing data.")), audit=AuditLog(path))
    agent.ask("How many venues?")
    agent.ask("How many venues?")          # a cache hit is still an audit event
    agent.ask("Which campaign won?")
    agent.ask("   ")                       # and so is a blank submission

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["outcome"] for r in records] == [
        "answered", "answered", "declined", "failed",
    ]
    assert [r["cached"] for r in records] == [False, True, False, False]
    assert len({r["request_id"] for r in records}) == 4


# --- column-level access control, end to end --------------------------------

def test_a_restricted_column_is_blocked_before_the_query_runs(db):
    agent = NLQAgent(db, Settings(restricted_columns="email"),
                     client=FakeModelClient(plan("SELECT email FROM customers")))
    result = agent.ask("List every customer email")

    assert result.outcome is Outcome.FAILED
    assert "restricted" in result.answer
    assert len(agent.client.requests) == 1, "no synthesis call on blocked SQL"


def test_the_model_is_told_which_columns_are_restricted(db):
    agent = NLQAgent(db, Settings(restricted_columns="email"),
                     client=FakeModelClient(plan("SELECT 1 AS n"), "ok"))
    assert "Restricted columns" in agent.system_prompt
    assert "email" in agent.system_prompt
    assert "Restricted columns" not in make_agent(db).system_prompt


def test_a_cache_hit_reports_no_cost(db):
    """A cost dashboard sums these fields; a hit that inherited the original
    usage would bill every repeat for calls that never happened."""
    agent = make_agent(db, plan("SELECT 1 AS n"), "ok")
    agent.ask("q")
    assert agent.ask("q").usage.calls == 0
