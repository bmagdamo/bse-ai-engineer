"""Read-only database access and schema introspection.

Three independent layers keep a generated query from doing damage:

1. The connection is opened read-only at the file level (`mode=ro`). This is
   the guarantee that matters -- it holds even if every other check is wrong.
2. A SQLite authorizer rejects everything except SELECT/READ/FUNCTION, which
   blocks ATTACH, PRAGMA, and writes inside the engine itself.
3. guards.validate() parses the SQL first, purely so failures are explainable.

Connections are **per thread**. Streamlit serves every session from one cached
`Database`, and `set_authorizer` is per-connection state: on a shared
connection one thread's `finally` could clear the authorizer while another was
mid-query, disabling layer 2 for that query. Sharing also turned 120
concurrent COUNT(*) queries from 0.35s into >140s.

The query timeout uses `Connection.interrupt()` from a timer thread rather
than a progress handler: interrupt() is designed for cross-thread use and
keeps a Python callback out of a loop that runs every few thousand opcodes.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from bse_nlq.errors import ConfigError, QueryExecutionError, QueryTimeoutError

log = logging.getLogger(__name__)

# Everything not listed is denied. SQLITE_RECURSIVE is the step a WITH
# RECURSIVE CTE takes; without it the engine answers a bare "not authorized"
# and the agent burns its repair attempt on SQL that was never wrong. It reads
# no data itself -- reads inside the CTE still go through SQLITE_READ.
_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}


def _authorizer(action, arg1, arg2, db_name, trigger):  # noqa: ARG001
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool
    elapsed_seconds: float

    @property
    def is_empty(self) -> bool:
        return not self.rows


@dataclass
class ColumnInfo:
    name: str
    type: str
    nullable: bool
    primary_key: bool


@dataclass
class TableInfo:
    name: str
    columns: list[ColumnInfo]
    foreign_keys: list[tuple[str, str, str]]  # (column, ref_table, ref_column)


class Database:
    """A read-only handle to the ticketing database."""

    def __init__(self, path: Path, max_rows: int = 200, timeout_seconds: float = 10.0):
        self.path = Path(path)
        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds
        if not self.path.exists():
            raise ConfigError(
                f"Database not found at {self.path}. Build it first: "
                "uv run python data/seed.py"
            )
        # No check_same_thread=False: each connection is only ever touched by
        # the thread that opened it, so we *want* sqlite3 to flag misuse.
        self._local = threading.local()
        self._lock = threading.Lock()
        # (owning thread, connection). The thread is kept so dead entries can
        # be pruned: Streamlit starts a script-runner thread per rerun, so an
        # append-only list would leak a file handle per interaction.
        self._connections: list[tuple[threading.Thread, sqlite3.Connection]] = []
        self._conn  # fail fast if the file is unreadable  # noqa: B018

    @property
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
            self._local.conn = conn
            with self._lock:
                self._prune_locked()
                self._connections.append((threading.current_thread(), conn))
            log.debug("Opened read-only connection on thread %s",
                      threading.current_thread().name)
        return conn

    def _prune_locked(self) -> None:
        """Drop connections whose owning thread has exited.

        References are released rather than closed: a sqlite3 connection
        cannot be closed from another thread, and CPython finalises it as soon
        as the last reference goes away.
        """
        self._connections = [
            (thread, conn) for thread, conn in self._connections if thread.is_alive()
        ]

    @property
    def live_connections(self) -> int:
        """Connections currently held open. For tests and diagnostics."""
        with self._lock:
            self._prune_locked()
            return len(self._connections)

    def close(self) -> None:
        """Close this thread's connections; release the rest to finalisation."""
        with self._lock:
            connections, self._connections = self._connections, []
        for thread, conn in connections:
            if thread is threading.current_thread():
                with suppress(sqlite3.Error):
                    conn.close()
        self._local = threading.local()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- execution --------------------------------------------------------

    def run_select(self, sql: str, timeout_seconds: float | None = None) -> QueryResult:
        """Execute a validated SELECT under the authorizer, row cap, and timeout.

        `timeout_seconds` overrides the configured budget for this call, which
        is how the agent hands down what is left of the question's end-to-end
        deadline: a query allowed its full 20s inside a budget with 3s left
        would blow the budget and report the wrong reason for it.
        """
        conn = self._conn
        budget = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        timed_out = threading.Event()

        def _abort() -> None:
            timed_out.set()
            conn.interrupt()      # thread-safe by design; aborts the running query

        # The authorizer is scoped to this call because introspection needs
        # PRAGMA, which it denies. That scoping is only safe because the
        # connection belongs to this thread alone.
        conn.set_authorizer(_authorizer)
        watchdog = threading.Timer(budget, _abort)
        watchdog.daemon = True
        watchdog.start()
        started = time.monotonic()
        try:
            cursor = conn.execute(sql)
            rows = cursor.fetchmany(self.max_rows + 1)
            columns = [d[0] for d in cursor.description] if cursor.description else []
        except sqlite3.Error as exc:
            if timed_out.is_set():
                raise QueryTimeoutError(
                    f"The query took longer than {budget:.0f}s and was stopped. "
                    "Try narrowing the question to a shorter time range."
                ) from exc
            log.warning("SQLite rejected the generated query: %s", exc)
            raise QueryExecutionError(
                "The database rejected the generated query.", str(exc), sql
            ) from exc
        finally:
            watchdog.cancel()
            conn.set_authorizer(None)

        elapsed = time.monotonic() - started
        if elapsed > 1.0:
            log.info("Slow query: %.2fs for %s", elapsed, sql.replace("\n", " ")[:120])
        # The extra row above the cap exists only to detect truncation. It
        # cannot fire when the caller already capped the SQL at exactly
        # max_rows (guards.enforce_limit does), so the agent reconstructs the
        # flag itself -- see NLQAgent._capped.
        return QueryResult(
            columns=columns,
            rows=rows[: self.max_rows],
            truncated=len(rows) > self.max_rows,
            elapsed_seconds=elapsed,
        )

    # -- introspection ----------------------------------------------------

    def tables(self) -> list[TableInfo]:
        """Read the live schema so prompt context can never drift from reality."""
        names = [
            r[0]
            for r in self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        return [
            TableInfo(
                name=name,
                columns=[
                    ColumnInfo(name=r[1], type=r[2] or "TEXT",
                               nullable=not r[3], primary_key=bool(r[5]))
                    for r in self._conn.execute(f"PRAGMA table_info('{name}')")
                ],
                foreign_keys=[
                    (r[3], r[2], r[4])
                    for r in self._conn.execute(f"PRAGMA foreign_key_list('{name}')")
                ],
            )
            for name in names
        ]

    def distinct_values(self, table: str, column: str, limit: int = 25) -> list[str]:
        """Sample distinct values for a low-cardinality column (prompt value hints)."""
        try:
            rows = self._conn.execute(
                f'SELECT DISTINCT "{column}" FROM "{table}" '
                f'WHERE "{column}" IS NOT NULL ORDER BY 1 LIMIT {int(limit)}'
            ).fetchall()
        except sqlite3.Error:
            return []
        return [str(r[0]) for r in rows]

    def row_count(self, table: str) -> int:
        return self._conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
