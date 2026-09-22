"""SQL validation and LIMIT enforcement.

The third of three safety layers. The real guarantees are in db.py (a
read-only connection plus a SQLite authorizer); this layer exists because a
parse-based check can say *what* was wrong, where SQLite only says
"attempt to write a readonly database".

Checks are a tuple of independent functions rather than an if-chain, so RULES
is the auditable list of what is enforced, each rule is testable alone, a
rejection names the rule that fired, and a caller can pass a different policy.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import suppress

import sqlglot
from sqlglot import exp

from bse_nlq.errors import UnsafeSQLError

log = logging.getLogger(__name__)

DIALECT = "sqlite"

#: Nodes that mutate data or schema. Matched across the whole tree, so a write
#: smuggled into a CTE or a subquery is caught too.
WRITE_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
    exp.TruncateTable, exp.Merge, exp.Grant, exp.Attach, exp.Detach,
    exp.Pragma, exp.Command,
)

READ_ROOTS: tuple[type[exp.Expression], ...] = (
    exp.Select, exp.Union, exp.Subquery, exp.With,
)

#: A rule returns the user-facing rejection message, or None to allow.
Rule = Callable[[list[exp.Expression]], str | None]


def single_statement(statements: list[exp.Expression]) -> str | None:
    if len(statements) > 1:
        return f"The generated SQL contained {len(statements)} statements; only one is allowed."
    return None


def read_only_root(statements: list[exp.Expression]) -> str | None:
    if not isinstance(statements[0], READ_ROOTS):
        return f"Only read-only SELECT queries are allowed (got {type(statements[0]).__name__})."
    return None


def no_write_operations(statements: list[exp.Expression]) -> str | None:
    for node in statements[0].walk():
        if isinstance(node, WRITE_NODES):
            return f"Only read-only SELECT queries are allowed (found {type(node).__name__})."
    return None


#: The audit surface: exactly what the guard enforces, in order.
RULES: tuple[Rule, ...] = (single_statement, read_only_root, no_write_operations)


def validate(sql: str, rules: tuple[Rule, ...] = RULES) -> exp.Expression:
    """Parse `sql` and apply every rule. Returns the AST, or raises."""
    text = (sql or "").strip().rstrip(";").strip()
    if not text:
        raise UnsafeSQLError("The model did not produce a query.")

    try:
        statements = [s for s in sqlglot.parse(text, read=DIALECT) if s is not None]
    except Exception as exc:  # sqlglot raises several parse error types
        raise UnsafeSQLError("The generated SQL could not be parsed.", str(exc)) from exc
    if not statements:
        raise UnsafeSQLError("The generated SQL could not be parsed.", "no statements found")

    for rule in rules:
        message = rule(statements)
        if message:
            # A rejection means the prompt let something through it shouldn't
            # have, so it is worth surfacing rather than silently retrying.
            log.warning("SQL rejected by rule %r: %s", rule.__name__, message)
            raise UnsafeSQLError(message, rule.__name__)

    return statements[0]


def _existing_limit(root: exp.Expression) -> int | None:
    """The row limit already in the query, when it is a plain integer literal.

    Anything else -- a placeholder, an expression -- reads as None so the
    caller replaces it rather than trusting a bound it cannot evaluate.
    """
    with suppress(AttributeError, TypeError, ValueError):
        return int(root.args["limit"].expression.this)
    return None


def enforce_limit(sql: str, max_rows: int) -> tuple[str, bool]:
    """Ensure the query returns at most `max_rows`.

    Returns (sql, limit_was_added). An existing smaller LIMIT is respected; an
    existing larger one is lowered. The flag matters downstream: only a limit
    we added means the result may be truncated -- a model-authored "LIMIT 10"
    for a top-10 question is a complete answer. See NLQAgent._capped.
    """
    root = validate(sql)
    current = _existing_limit(root)
    if current is not None and current <= max_rows:
        return root.sql(dialect=DIALECT, pretty=True), False

    log.debug("Injected LIMIT %d into the generated query", max_rows)
    return root.limit(max_rows).sql(dialect=DIALECT, pretty=True), True
