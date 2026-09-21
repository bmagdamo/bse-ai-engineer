"""Shared fixtures.

The database is session-scoped and read-only, so every test can share one
connection.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bse_nlq.config import SETTINGS
from bse_nlq.db import Database


@pytest.fixture(scope="session")
def db_path() -> Path:
    path = SETTINGS.db_path
    if not path.exists():
        pytest.skip(f"database not built at {path}; run: uv run python data/seed.py")
    return path


@pytest.fixture(scope="session")
def db(db_path: Path):
    with Database(db_path, max_rows=50, timeout_seconds=10) as handle:
        yield handle
