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
from collections.abc import Callable, Sequence
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


def restricted_columns(names: Sequence[str]) -> Rule:
    """Build a rule rejecting any query that reads one of `names`.

    Matching is by column *name*, not by table-qualified name: resolving an
    alias back to the table it came from needs full scope analysis, and for an
    access rule the cheap check errs in the safe direction -- it over-blocks
    rather than letting a restricted column through under an alias the
    resolver did not follow.

    A bare `*` is rejected for the same reason: it expands to whatever the
    table holds, restricted columns included, so a wildcard would be the way
    around the rule. `COUNT(*)` is untouched -- it names no column and returns
    no data -- which is why the check is on the parent node rather than on
    Star itself.

    Aliasing to a restricted name (`SELECT city AS email`) is not blocked: the
    rule governs which data is read, and that query reads `city`.
    """
    denied = frozenset(name.strip().casefold() for name in names if name.strip())

    def rule(statements: list[exp.Expression]) -> str | None:
        for node in statements[0].walk():
            if isinstance(node, exp.Star) and not isinstance(node.parent, exp.Func):
                return (
                    "SELECT * is not allowed while column restrictions are in force. "
                    "Name the columns you need instead."
                )
            if isinstance(node, exp.Column) and node.name.casefold() in denied:
                return (
                    f"Column '{node.name}' is restricted and cannot be queried. "
                    "Ask for an aggregate or a non-restricted column instead."
                )
        return None

    rule.__name__ = "restricted_columns"   # rejections are logged by rule name
    return rule


#: The audit surface: exactly what the guard enforces, in order.
RULES: tuple[Rule, ...] = (single_statement, read_only_root, no_write_operations)


def policy_for(restricted: Sequence[str] = ()) -> tuple[Rule, ...]:
    """The rule set for a given access policy.

    Composed rather than configured: the baseline rules are constant, and an
    installation that restricts columns simply gets one more rule on the end.
    With nothing restricted the tuple is `RULES` itself, so the default path
    carries no extra work and no extra branch.
    """
    if not any(name.strip() for name in restricted):
        return RULES
    return (*RULES, restricted_columns(restricted))


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

    Anything else -- a placeholder, an expression, or a root that cannot
    carry a LIMIT at all -- reads as None so the caller replaces it rather
    than trusting a bound it cannot evaluate.

    .get, not [], because only a Select carries a "limit" key unconditionally:
    on the other READ_ROOTS (a UNION, a parenthesised subquery) the key is
    absent until a LIMIT is actually parsed, and a KeyError here escapes the
    suppression and every handler above it.
    """
    with suppress(AttributeError, TypeError, ValueError):
        return int(root.args.get("limit").expression.this)
    return None


def enforce_limit(sql: str, max_rows: int, rules: tuple[Rule, ...] = RULES
                  ) -> tuple[str, bool]:
    """Ensure the query returns at most `max_rows`.

    Returns (sql, limit_was_added). An existing smaller LIMIT is respected; an
    existing larger one is lowered. The flag matters downstream: only a limit
    we added means the result may be truncated -- a model-authored "LIMIT 10"
    for a top-10 question is a complete answer. See NLQAgent._capped.
    """
    root = validate(sql, rules)
    current = _existing_limit(root)
    if current is not None and current <= max_rows:
        return root.sql(dialect=DIALECT, pretty=True), False

    log.debug("Injected LIMIT %d into the generated query", max_rows)
    return root.limit(max_rows).sql(dialect=DIALECT, pretty=True), True
