"""Turn the live database schema into prompt context.

Introspected rather than hardcoded, so the prompt can never drift away from
the actual schema.

The value hints are the highest-leverage part of this module: telling the
model that orders.status is one of {completed, refunded, cancelled} prevents
the single most common text-to-SQL failure -- filtering on a plausible but
non-existent literal like 'complete'.
"""

from __future__ import annotations

from bse_nlq.db import Database
from bse_nlq.errors import ConfigError

# Columns whose full value set is small enough to enumerate in the prompt.
VALUE_HINT_COLUMNS: list[tuple[str, str]] = [
    ("orders", "status"),
    ("orders", "channel"),
    ("events", "status"),
    ("event_categories", "name"),
    ("teams", "name"),
    ("teams", "league"),
    ("venues", "name"),
    ("seating_sections", "level"),
    ("customers", "loyalty_tier"),
]


def render_schema(db: Database) -> str:
    """A compact CREATE-TABLE-style rendering of every table."""
    blocks = []
    for table in db.tables():
        lines = [f"TABLE {table.name} ({db.row_count(table.name):,} rows)"]
        for col in table.columns:
            flags = []
            if col.primary_key:
                flags.append("PK")
            if not col.nullable:
                flags.append("NOT NULL")
            suffix = f"  -- {', '.join(flags)}" if flags else ""
            lines.append(f"    {col.name} {col.type}{suffix}")
        for column, ref_table, ref_column in table.foreign_keys:
            lines.append(f"    FOREIGN KEY {column} -> {ref_table}.{ref_column}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_value_hints(db: Database) -> str:
    """Enumerate the allowed literals for low-cardinality columns.

    VALUE_HINT_COLUMNS is hand-picked rather than auto-detected: probing every
    text column for cardinality would mean a COUNT(DISTINCT) over the 500k-row
    tickets table at every startup. The tradeoff is that the list can drift
    from the schema, so it is validated against the live database here and a
    stale entry fails loudly instead of silently dropping a hint the prompt
    depends on.
    """
    known = {table.name: {c.name for c in table.columns} for table in db.tables()}
    missing = [
        f"{table}.{column}"
        for table, column in VALUE_HINT_COLUMNS
        if column not in known.get(table, set())
    ]
    if missing:
        raise ConfigError(
            "VALUE_HINT_COLUMNS refers to columns that no longer exist: "
            f"{', '.join(missing)}. Update schema_context.py to match the schema."
        )

    lines = []
    for table, column in VALUE_HINT_COLUMNS:
        values = db.distinct_values(table, column)
        if not values:
            continue
        rendered = ", ".join(f"'{v}'" for v in values)
        lines.append(f"  {table}.{column}: {rendered}")
    return "\n".join(lines)


def render_date_range(db: Database) -> str:
    """Tell the model what period the data actually covers.

    Without this it will happily write a query for 2019 and report zero as if
    it were a real business answer.
    """
    result = db.run_select("SELECT MIN(event_date), MAX(event_date) FROM events")
    if not result.rows:
        return "unknown"
    lo, hi = result.rows[0]
    return f"{lo} to {hi}"


def build_schema_context(db: Database) -> str:
    """The full, byte-stable schema block injected into the system prompt."""
    return (
        "## Database schema (SQLite)\n\n"
        f"{render_schema(db)}\n\n"
        "## Allowed values for low-cardinality columns\n\n"
        "Use these literals exactly -- do not guess variants.\n\n"
        f"{render_value_hints(db)}\n\n"
        "## Data coverage\n\n"
        f"  events.event_date spans {render_date_range(db)}.\n"
        "  Questions about dates outside this range will correctly return no rows.\n"
    )
