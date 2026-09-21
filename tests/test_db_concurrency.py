"""Thread safety of the read-only Database.

Streamlit serves every session from one @st.cache_resource Database, so a
single shared connection was both a correctness bug (one thread's `finally`
clearing the authorizer mid-query on another) and a performance one (120
concurrent queries went from 0.35s to over 140s). These tests pin the fix.
"""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest

from bse_nlq.db import Database
from bse_nlq.errors import QueryExecutionError, QueryTimeoutError


@pytest.fixture
def fresh_db(db_path):
    handle = Database(db_path, max_rows=50, timeout_seconds=10)
    yield handle
    handle.close()


def _run_on_threads(fn, count: int):
    results, errors = [], []

    def worker():
        try:
            fn(results)
        except Exception as exc:  # noqa: BLE001 - the test reports whatever escapes
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=worker) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results, errors


def test_concurrent_queries_succeed_and_stay_fast(fresh_db):
    started = time.perf_counter()
    results, errors = _run_on_threads(
        lambda out: [out.append(fresh_db.run_select("SELECT COUNT(*) FROM tickets").rows[0][0])
                     for _ in range(10)],
        count=8,
    )
    elapsed = time.perf_counter() - started

    assert errors == []
    assert len(results) == 80
    assert len(set(results)) == 1, "every thread must see the same committed data"
    # Generous bound: the shared-connection version took >140s for this shape.
    assert elapsed < 15, f"concurrent queries degraded badly ({elapsed:.1f}s)"


def test_each_thread_gets_its_own_connection(fresh_db):
    seen: set[int] = set()
    lock = threading.Lock()

    def record(_out):
        conn = fresh_db._conn
        with lock:
            seen.add(id(conn))

    _run_on_threads(record, count=4)
    assert len(seen) == 4, "connections must not be shared across threads"


def test_a_thread_reuses_its_own_connection(fresh_db):
    assert fresh_db._conn is fresh_db._conn


def test_authorizer_is_enforced_on_every_thread(fresh_db):
    """The race being fixed: one thread clearing the authorizer while another
    is mid-query would let a write through layer 2."""
    def attempt(out):
        for _ in range(5):
            try:
                fresh_db.run_select("INSERT INTO venues VALUES (99,'x','y','z',1)")
                out.append("ALLOWED")
            except (QueryExecutionError, sqlite3.Error):
                out.append("blocked")

    results, errors = _run_on_threads(attempt, count=6)
    assert errors == []
    assert set(results) == {"blocked"}, "a write escaped the authorizer under concurrency"


def test_close_releases_every_connection(fresh_db):
    _run_on_threads(lambda out: out.append(fresh_db.run_select("SELECT 1").rows), count=3)
    fresh_db.close()
    assert fresh_db._connections == []


def test_connections_from_dead_threads_are_pruned(fresh_db):
    """Streamlit starts a new script-runner thread per rerun, so without
    pruning the pool would leak one file handle per user interaction."""
    for _ in range(12):
        _run_on_threads(lambda out: out.append(fresh_db.run_select("SELECT 1").rows), count=1)
    # Only the main thread's connection should survive; the 12 short-lived
    # worker threads have exited.
    assert fresh_db.live_connections <= 2, (
        f"connection pool grew to {fresh_db.live_connections}; dead threads are leaking"
    )


def test_close_is_idempotent(fresh_db):
    fresh_db.close()
    fresh_db.close()


def test_timeout_aborts_a_runaway_query(db_path):
    with Database(db_path, max_rows=10, timeout_seconds=1.0) as handle:
        started = time.perf_counter()
        with pytest.raises(QueryTimeoutError):
            handle.run_select("SELECT COUNT(*) FROM tickets a, tickets b, tickets c")
        assert time.perf_counter() - started < 6, "interrupt() did not fire promptly"


def test_connection_is_reusable_after_a_timeout(db_path):
    """interrupt() must abort the statement, not poison the connection."""
    with Database(db_path, max_rows=10, timeout_seconds=1.0) as handle:
        with pytest.raises(QueryTimeoutError):
            handle.run_select("SELECT COUNT(*) FROM tickets a, tickets b, tickets c")
        assert handle.run_select("SELECT COUNT(*) FROM venues").rows == [(2,)]


def test_watchdog_does_not_fire_on_a_fast_query(fresh_db):
    """A cancelled timer must not interrupt the next statement."""
    for _ in range(20):
        assert fresh_db.run_select("SELECT COUNT(*) FROM venues").rows == [(2,)]
