"""Runtime settings. Everything tunable lives here, not scattered in the code."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import find_dotenv, load_dotenv

from bse_nlq.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_env() -> Path | None:
    """Load .env before any setting is read.

    Walking up from the cwd first lets a developer keep a local override
    anywhere; the repo-root fallback is what makes `nlq` work when invoked
    from outside the project, where a bare load_dotenv() would find nothing
    and the agent would fail with "no API key" despite the file existing.

    Real environment variables always win over the file.
    """
    for candidate in (find_dotenv(usecwd=True), REPO_ROOT / ".env"):
        path = Path(candidate) if candidate else None
        if path and path.is_file():
            load_dotenv(path)          # override=False: real env vars win
            return path
    return None


DOTENV_PATH = _load_env()


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable configuration.

    Defaults are plain literals, never os.getenv(): reading the environment in
    a field default binds it once at import, which makes the value invisible
    to tests and dependent on import order. `from_env()` reads the
    environment, so `Settings()` alone stays pure and predictable.
    """

    # --- model ---------------------------------------------------------
    # claude-opus-5 is the default; see README "Model selection".
    model: str = "claude-opus-5"
    # Effort for the NL -> SQL step: a deliberate latency/cost choice for an
    # interactive CLI. Raise to "high" for a harder schema.
    sql_effort: str = "medium"
    # Answer synthesis is a formatting task -> cheapest useful setting.
    answer_effort: str = "low"
    # Thinking is ON by default on Opus 5 and max_tokens caps thinking +
    # output together, so this needs real headroom.
    sql_max_tokens: int = 8_000
    answer_max_tokens: int = 2_000

    # --- database ------------------------------------------------------
    db_path: Path = REPO_ROOT / "data" / "bse.db"
    max_rows: int = 200
    # The business calendar. "Last month" is resolved in this zone and passed
    # to the model as a literal date, because SQLite's date('now') is UTC and
    # would disagree with it for part of every day. See prompts.DATE_RECIPES.
    timezone: str = "America/New_York"
    # Aggregates over ~850k ticket rows take a second or two; this is a
    # runaway-query backstop, not a latency target.
    query_timeout_seconds: float = 20.0

    # --- retries -------------------------------------------------------
    # Tenacity owns the retry policy; the SDK's own retries are disabled so
    # attempts are not multiplied (3 x 3 = 9 real calls). See claude.py.
    max_attempts: int = 3
    retry_initial_seconds: float = 1.0
    retry_max_seconds: float = 10.0
    # A server-sent `retry-after` is honoured in full rather than clamped to
    # retry_max_seconds -- waiting less than the service asked for is how you
    # get rate-limited again immediately. This is the ceiling on trusting it,
    # so a pathological header cannot hang an interactive session.
    retry_after_max_seconds: float = 60.0
    request_timeout_seconds: float = 120.0

    # --- agent ---------------------------------------------------------
    # One repair attempt: enough to fix a mistyped column, not enough to burn
    # budget looping on a fundamentally wrong query.
    max_repair_attempts: int = 1

    @classmethod
    def from_env(cls, **overrides) -> Settings:
        """Build settings from NLQ_* environment variables, then apply
        overrides (CLI flags), which always win.

        Fields and their casters are derived from the dataclass rather than
        listed by hand: a hand-kept list silently ignores any field someone
        forgets to add to it.
        """
        env = {}
        for name, caster in _CASTERS.items():
            raw = (os.getenv(f"NLQ_{name.upper()}") or "").strip()
            if not raw:
                continue
            try:
                env[name] = caster(raw)
            except ValueError as exc:
                raise ConfigError(
                    f"NLQ_{name.upper()} is not a valid {caster.__name__}: {raw!r}"
                ) from exc
        return cls(**env | {k: v for k, v in overrides.items() if v is not None})

    def __post_init__(self) -> None:
        """Reject values that would otherwise fail far from their cause: a
        negative max_repair_attempts made the repair loop iterate zero times
        and hit its "unreachable" assertion, and max_rows below 1 produced a
        LIMIT 0. Failing here names the setting instead."""
        for name, floor in _FLOORS.items():
            if getattr(self, name) < floor:
                raise ConfigError(
                    f"NLQ_{name.upper()} must be >= {floor}, got {getattr(self, name)!r}."
                )
        try:
            ZoneInfo(self.timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ConfigError(
                f"NLQ_TIMEZONE is not a known IANA timezone: {self.timezone!r}."
            ) from exc

    # --- preconditions -------------------------------------------------

    def require_api_key(self) -> str:
        key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not key:
            where = f"loaded {DOTENV_PATH}" if DOTENV_PATH else "no .env file found"
            raise ConfigError(
                "ANTHROPIC_API_KEY is not set. Create a .env file in the project root "
                "with ANTHROPIC_API_KEY=sk-ant-... (start from .env.example), or export "
                f"it into your shell. [{where}]"
            )
        return key

    def require_database(self) -> Path:
        if not self.db_path.exists():
            raise ConfigError(
                f"Database not found at {self.db_path}. Build it first: "
                "uv run python data/seed.py"
            )
        return self.db_path


#: Minimum sensible value per field; anything lower fails at construction.
_FLOORS = {
    "max_rows": 1, "max_attempts": 1, "max_repair_attempts": 0,
    "query_timeout_seconds": 0, "sql_max_tokens": 1, "answer_max_tokens": 1,
}

#: field -> caster, derived from the annotations so the two cannot drift.
#: `f.type` is the annotation *string* because of `from __future__ import
#: annotations`. A field of a new type raises KeyError here at import, which
#: is the point: adding one forces a decision about how to parse it.
_TYPES = {"str": str, "int": int, "float": float, "Path": Path}
_CASTERS = {f.name: _TYPES[f.type] for f in fields(Settings)}

#: Process-wide settings, resolved once from the environment at import.
SETTINGS = Settings.from_env()
