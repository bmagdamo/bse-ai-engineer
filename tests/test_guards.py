"""The SQL guard must reject anything that is not a single read-only SELECT."""

from __future__ import annotations

import logging

import pytest

from bse_nlq.errors import UnsafeSQLError
from bse_nlq.guards import enforce_limit, validate

SAFE = [
    "SELECT 1",
    "SELECT * FROM events LIMIT 10",
    "SELECT c.name, SUM(t.price_paid) FROM tickets t JOIN events e ON e.event_id=t.event_id "
    "JOIN event_categories c ON c.category_id=e.category_id GROUP BY c.name",
    "WITH recent AS (SELECT * FROM events WHERE event_date > '2025-01-01') "
    "SELECT COUNT(*) FROM recent",
    "SELECT * FROM events WHERE event_id IN (SELECT event_id FROM tickets)",
    "SELECT 1 UNION SELECT 2",
    "SELECT * FROM events;",           # trailing semicolon is fine
]

UNSAFE = [
    ("DROP TABLE events", "drop"),
    ("DELETE FROM tickets", "delete"),
    ("INSERT INTO venues VALUES (3,'x','y','z',1)", "insert"),
    ("UPDATE orders SET status='completed'", "update"),
    ("CREATE TABLE evil (id INTEGER)", "create"),
    ("ALTER TABLE events ADD COLUMN evil TEXT", "alter"),
    ("ATTACH DATABASE '/tmp/evil.db' AS evil", "attach"),
    ("PRAGMA table_info('events')", "pragma"),
    ("SELECT 1; DROP TABLE events", "stacked statements"),
    ("SELECT 1; DELETE FROM tickets;", "stacked statements"),
    ("SELECT 1 -- harmless\n; DROP TABLE events", "comment-smuggled DML"),
    ("", "empty"),
    ("   ", "whitespace only"),
    ("this is not sql at all !!", "unparseable"),
]


@pytest.mark.parametrize("sql", SAFE)
def test_accepts_read_only_selects(sql):
    assert validate(sql) is not None


@pytest.mark.parametrize("sql", [pytest.param(sql, id=label) for sql, label in UNSAFE])
def test_rejects_non_select(sql):
    with pytest.raises(UnsafeSQLError):
        validate(sql)


def test_adds_limit_when_missing():
    sql, added = enforce_limit("SELECT * FROM events", max_rows=50)
    assert added is True
    assert "LIMIT 50" in sql.upper()


def test_respects_smaller_existing_limit():
    sql, added = enforce_limit("SELECT * FROM events LIMIT 5", max_rows=50)
    assert added is False
    assert "LIMIT 5" in sql.upper()
    assert "LIMIT 50" not in sql.upper()


def test_lowers_oversized_existing_limit():
    sql, _ = enforce_limit("SELECT * FROM events LIMIT 100000", max_rows=50)
    assert "LIMIT 50" in sql.upper()
    assert "100000" not in sql


def test_enforce_limit_still_rejects_unsafe_sql():
    with pytest.raises(UnsafeSQLError):
        enforce_limit("DROP TABLE events", max_rows=50)


# --- the rule set is an auditable composite --------------------------------

def test_rules_are_enumerable_and_usable_in_isolation():
    import sqlglot

    from bse_nlq.guards import RULES, no_write_operations, read_only_root, single_statement

    assert (single_statement, read_only_root, no_write_operations) == RULES

    def parse(sql):
        return sqlglot.parse(sql, read="sqlite")

    assert single_statement(parse("SELECT 1")) is None
    assert single_statement(parse("SELECT 1; SELECT 2"))
    assert read_only_root(parse("SELECT 1")) is None
    assert read_only_root(parse("DROP TABLE events"))
    assert no_write_operations(parse("SELECT 1")) is None


def test_a_custom_rule_set_can_be_injected():
    """Rules are a parameter, so a caller can tighten or relax the policy
    without editing the guard."""
    from bse_nlq.guards import validate

    with pytest.raises(UnsafeSQLError, match="Nothing is allowed"):
        validate("SELECT 1", rules=(lambda _statements: "Nothing is allowed here.",))


def test_rejections_are_logged_with_the_rule_name(caplog):
    with caplog.at_level(logging.WARNING, logger="bse_nlq.guards"), \
            pytest.raises(UnsafeSQLError):
        validate("DROP TABLE events")
    assert any("read_only_root" in r.getMessage() for r in caplog.records)
