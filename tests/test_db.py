"""The database handle must be read-only and must cap runaway result sets."""

from __future__ import annotations

import sqlite3

import pytest

from bse_nlq.db import Database
from bse_nlq.errors import ConfigError, QueryExecutionError

EXPECTED_TABLES = {
    "venues", "teams", "event_categories", "events",
    "seating_sections", "customers", "orders", "tickets",
}


def test_missing_database_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError):
        Database(tmp_path / "nope.db")


def test_select_returns_columns_and_rows(db):
    result = db.run_select("SELECT name, city FROM venues ORDER BY venue_id")
    assert result.columns == ["name", "city"]
    assert ("Barclays Center", "Brooklyn") in result.rows


def test_row_cap_is_enforced_and_flagged(db):
    result = db.run_select("SELECT event_id FROM events")
    assert len(result.rows) <= db.max_rows
    assert result.truncated is True


def test_empty_result_is_not_an_error(db):
    result = db.run_select("SELECT * FROM events WHERE event_date = '1900-01-01'")
    assert result.is_empty
    assert result.rows == []


def test_writes_are_impossible(db):
    # The file is opened mode=ro, so this fails at the SQLite layer even
    # before any of our own validation runs.
    with pytest.raises((QueryExecutionError, sqlite3.Error)):
        db.run_select("INSERT INTO venues VALUES (99,'x','y','z',1)")


def test_bad_column_surfaces_the_sqlite_message(db):
    with pytest.raises(QueryExecutionError) as excinfo:
        db.run_select("SELECT no_such_column FROM events")
    # The raw message is what the repair loop feeds back to the model.
    assert "no_such_column" in excinfo.value.sqlite_message


def test_introspection_sees_every_table(db):
    assert {t.name for t in db.tables()} >= EXPECTED_TABLES


def test_introspection_reads_foreign_keys(db):
    events = next(t for t in db.tables() if t.name == "events")
    refs = {(c, rt) for c, rt, _ in events.foreign_keys}
    assert ("venue_id", "venues") in refs
    assert ("home_team_id", "teams") in refs
