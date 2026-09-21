"""Result rendering: for the model (markdown) and for the terminal (rich)."""

from __future__ import annotations

from rich.table import Table

from bse_nlq.db import QueryResult

MAX_ROWS_FOR_MODEL = 50
MAX_CELL_CHARS = 60


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    text = str(value)
    return text if len(text) <= MAX_CELL_CHARS else text[: MAX_CELL_CHARS - 1] + "…"


def to_markdown(result: QueryResult, max_rows: int = MAX_ROWS_FOR_MODEL) -> str:
    """Render rows as a markdown table for the answer-synthesis prompt.

    Capped independently of the display cap: the synthesis step only needs
    enough rows to describe the shape of the answer, not all of them.
    """
    if not result.rows:
        return "(no rows)"

    rows = result.rows[:max_rows]
    header = "| " + " | ".join(result.columns) + " |"
    divider = "| " + " | ".join("---" for _ in result.columns) + " |"
    body = ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    out = "\n".join([header, divider, *body])
    if len(result.rows) > max_rows:
        out += f"\n\n(showing {max_rows} of {len(result.rows)} returned rows)"
    return out


def to_rich_table(result: QueryResult, max_rows: int = 25) -> Table:
    """Render rows as a terminal table."""
    table = Table(show_header=True, header_style="bold cyan", box=None, pad_edge=False)
    for column in result.columns:
        table.add_column(column, overflow="fold")
    for row in result.rows[:max_rows]:
        table.add_row(*(_cell(v) for v in row))
    return table


def to_dicts(result: QueryResult) -> list[dict]:
    """Row dicts, for the Streamlit dataframe."""
    # strict=True: a row that does not line up with the header is a bug in
    # the DB layer, not something to silently truncate.
    return [dict(zip(result.columns, row, strict=True)) for row in result.rows]
