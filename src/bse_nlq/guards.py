"""SQL validation and LIMIT enforcement.

This is the *third* safety layer. The real guarantees live in db.py (a
read-only connection plus a SQLite authorizer). This layer exists because a
parse-based check can explain precisely what was wrong, whereas SQLite just
raises "attempt to write a readonly database".

The checks are a composite of independent `SqlRule` objects rather than an
if-chain. For a security boundary that matters: the rule set is enumerable
(`RULES`), each rule is testable in isolation, a rejection names the rule that
fired, and adding a rule does not mean editing the function every other rule
flows through.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import sqlglot
from sqlglot import exp

from bse_nlq.errors import UnsafeSQLError

log = logging.getLogger(__name__)

DIALECT = "sqlite"

#: Node types that mutate data or schema. Checked across the whole tree, so a
#: write smuggled into a CTE or a subquery is caught too.
WRITE_NODES: tuple[type[exp.Expression], ...] = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
    exp.TruncateTable, exp.Merge, exp.Grant, exp.Attach, exp.Detach,
    exp.Pragma, exp.Command,
)

#: Node types permitted at the root of a statement.
READ_ROOTS: tuple[type[exp.Expression], ...] = (
    exp.Select, exp.Union, exp.Subquery, exp.With,
)


@dataclass(frozen=True, slots=True)
class Violation:
    """Why a statement was rejected, and by which rule."""

    rule: str
    message: str
    detail: str = ""


@runtime_checkable
class SqlRule(Protocol):
    """One independent check over the parsed statements."""

    name: str

    def check(self, statements: list[exp.Expression]) -> Violation | None: ...


@dataclass(frozen=True, slots=True)
class SingleStatement:
    name: str = "single_statement"

    def check(self, statements: list[exp.Expression]) -> Violation | None:
        if len(statements) == 1:
            return None
        return Violation(
            self.name,
            "The generated SQL contained more than one statement, which is not allowed.",
            f"found {len(statements)} statements",
        )


@dataclass(frozen=True, slots=True)
class ReadOnlyRoot:
    name: str = "read_only_root"

    def check(self, statements: list[exp.Expression]) -> Violation | None:
        root = statements[0]
        if isinstance(root, READ_ROOTS):
            return None
        return Violation(
            self.name,
            "Only read-only SELECT queries are allowed.",
            f"top-level node was {type(root).__name__}",
        )


@dataclass(frozen=True, slots=True)
class NoWriteOperations:
    name: str = "no_write_operations"

    def check(self, statements: list[exp.Expression]) -> Violation | None:
        for node in statements[0].walk():
            if isinstance(node, WRITE_NODES):
                return Violation(
                    self.name,
                    "Only read-only SELECT queries are allowed.",
                    f"found a {type(node).__name__} clause",
                )
        return None


#: The full rule set, applied in order. Enumerable on purpose -- this tuple is
#: the audit surface for what the guard actually enforces.
RULES: tuple[SqlRule, ...] = (SingleStatement(), ReadOnlyRoot(), NoWriteOperations())


def validate(sql: str, rules: tuple[SqlRule, ...] = RULES) -> exp.Expression:
    """Parse `sql` and apply every rule. Returns the AST, or raises."""
    text = (sql or "").strip().rstrip(";").strip()
    if not text:
        raise UnsafeSQLError("The model did not produce a query.")

    try:
        parsed = sqlglot.parse(text, read=DIALECT)
    except Exception as exc:  # sqlglot raises several parse error types
        raise UnsafeSQLError("The generated SQL could not be parsed.", str(exc)) from exc

    statements = [statement for statement in parsed if statement is not None]
    if not statements:
        raise UnsafeSQLError("The generated SQL could not be parsed.", "no statements found")

    for rule in rules:
        violation = rule.check(statements)
        if violation is not None:
            # A rejection means the prompt let something through it shouldn't
            # have, so it is worth surfacing rather than silently retrying.
            log.warning("SQL rejected by rule %r: %s", violation.rule, violation.detail)
            raise UnsafeSQLError(violation.message, violation.detail)

    return statements[0]


def enforce_limit(sql: str, max_rows: int) -> tuple[str, bool]:
    """Ensure the query returns at most `max_rows`.

    Returns (sql, limit_was_added). An existing smaller LIMIT is respected; an
    existing larger one is lowered.
    """
    root = validate(sql)

    existing = root.args.get("limit")
    if existing is not None:
        try:
            current = int(existing.expression.this)
        except (AttributeError, TypeError, ValueError):
            current = None
        if current is not None and current <= max_rows:
            return root.sql(dialect=DIALECT, pretty=True), False

    log.debug("Injected LIMIT %d into the generated query", max_rows)
    return root.limit(max_rows).sql(dialect=DIALECT, pretty=True), True


def prettify(sql: str) -> str:
    """Best-effort formatting for display. Never raises."""
    try:
        return sqlglot.transpile(sql, read=DIALECT, write=DIALECT, pretty=True)[0]
    except Exception:
        return sql.strip()
