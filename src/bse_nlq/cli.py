"""Command-line interface.

Presentation only: it parses arguments, calls NLQAgent.ask(), and renders.
All behaviour lives in the agent, which is why app.py can be a thin wrapper
with no duplicated logic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax

from bse_nlq import __version__, formatter
from bse_nlq.agent import NLQAgent, NLQResult, Outcome
from bse_nlq.config import Settings
from bse_nlq.db import Database
from bse_nlq.errors import NLQError
from bse_nlq.logging_setup import configure as configure_logging
from bse_nlq.prompts import EXAMPLE_QUESTIONS

console = Console()

MAX_DISPLAY_ROWS = 25

#: Panel colour per outcome. Declining is yellow, not red: it is correct
#: behaviour, not an error.
_OUTCOME_STYLE = {
    Outcome.ANSWERED: "green",
    Outcome.DECLINED: "yellow",
    Outcome.FAILED: "red",
}

BANNER = """[bold]BSE Natural Language Query agent[/bold]
Ask about events, ticket sales, revenue, venues, and customers in plain English.
Type [cyan]exit[/cyan] to quit."""


# -- rendering ---------------------------------------------------------------
# Model answers and SQLite errors are arbitrary text: escape() everything
# dynamic, or rich parses a stray "[...]" as a markup tag and drops it.

def _render_answer(result: NLQResult) -> None:
    console.print(Panel(escape(result.answer), title="Answer", title_align="left",
                        border_style=_OUTCOME_STYLE[result.outcome]))


def _render_sql(result: NLQResult) -> None:
    if not result.sql:
        return
    console.print("\n[bold]Generated SQL[/bold]")
    console.print(Syntax(result.sql, "sql", theme="ansi_dark", word_wrap=True))
    if result.repaired:
        first_error = next(a.error for a in result.attempts if a.error)
        console.print(f"\n[dim]First attempt failed ({escape(first_error)}); retried once.[/dim]")


def _render_rows(result: NLQResult) -> None:
    if not result.rows:
        return
    console.print(f"\n[bold]Results[/bold] [dim]({result.row_count} rows, "
                  f"{result.elapsed_seconds:.2f}s)[/dim]")
    console.print(formatter.to_rich_table(result, MAX_DISPLAY_ROWS))
    if result.truncated:
        console.print("[dim]…truncated at the row cap; more rows exist.[/dim]")
    elif result.row_count > MAX_DISPLAY_ROWS:
        console.print(f"[dim]…showing first {MAX_DISPLAY_ROWS} of {result.row_count} rows.[/dim]")


def _render_footer(result: NLQResult) -> None:
    if result.assumptions:
        console.print("\n[bold]Assumptions[/bold]")
        for item in result.assumptions:
            console.print(f"  [dim]•[/dim] {escape(item)}")
    if result.usage.calls:
        console.print(f"\n[dim]{escape(result.usage.summary())}[/dim]")
    console.print()


def render(result: NLQResult, show_sql_only: bool = False) -> None:
    if show_sql_only:
        console.print(Syntax(result.sql or "-- no query generated", "sql", theme="ansi_dark"))
        return
    _render_answer(result)
    _render_sql(result)
    _render_rows(result)
    _render_footer(result)


# -- modes -------------------------------------------------------------------

def run_repl(agent: NLQAgent, show_sql_only: bool) -> int:
    console.print(Panel(BANNER, border_style="cyan"))
    while True:
        try:
            question = console.input("[bold cyan]?[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye.")
            return 0
        if question.lower() in {"exit", "quit", ":q"}:
            console.print("Bye.")
            return 0
        if not question:
            continue
        with console.status("[dim]Thinking…[/dim]"):
            result = agent.ask(question)
        render(result, show_sql_only)


def run_demo(agent: NLQAgent, show_sql_only: bool) -> int:
    """Exit non-zero if any question errored, so --demo works as a smoke test."""
    failures = 0
    for question in EXAMPLE_QUESTIONS:
        console.rule(f"[bold]{escape(question)}")
        result = agent.ask(question)
        failures += not result.ok
        render(result, show_sql_only)
    return 1 if failures else 0


def run_once(agent: NLQAgent, question: str, show_sql_only: bool) -> int:
    """One-shot mode is scriptable, so failure must be visible in the exit code."""
    result = agent.ask(question)
    render(result, show_sql_only)
    return 0 if result.ok else 1


# -- entry point -------------------------------------------------------------

def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="nlq",
        description="Ask questions about the BSE ticketing database in plain English.",
    )
    parser.add_argument("-q", "--question", help="Ask one question and exit.")
    parser.add_argument("--demo", action="store_true",
                        help="Run the example questions from the exercise brief.")
    parser.add_argument("--show-sql-only", action="store_true",
                        help="Print only the generated SQL (useful for piping).")
    parser.add_argument("--db", type=Path, dest="db_path",
                        help="Path to the SQLite database.")
    parser.add_argument("--model", help="Override the Claude model id.")
    parser.add_argument("--version", action="version", version=f"nlq {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show debug logging (retries, repairs, slow queries).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_logging(args.verbose)
    # CLI flags override the environment; None values are ignored.
    settings = Settings.from_env(db_path=args.db_path, model=args.model)

    try:
        settings.require_database()
        settings.require_api_key()
        with Database(settings.db_path, settings.max_rows,
                      settings.query_timeout_seconds) as db:
            agent = NLQAgent(db, settings)
            if args.demo:
                return run_demo(agent, args.show_sql_only)
            if args.question:
                return run_once(agent, args.question, args.show_sql_only)
            return run_repl(agent, args.show_sql_only)
    except NLQError as exc:
        console.print(f"[red]{escape(exc.user_message)}[/red]")
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
