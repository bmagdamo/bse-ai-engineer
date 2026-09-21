"""Runtime settings. Everything tunable lives here, not scattered in the code."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from bse_nlq.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_env() -> Path | None:
    """Load .env before any setting is read.

    Prefer a .env found by walking up from the current working directory (so a
    developer can keep a local override wherever they are), then fall back to
    the one next to the repo. The fallback is what makes `nlq` work when it is
    invoked from outside the project directory -- a bare load_dotenv() would
    silently find nothing and the agent would fail with "no API key" even
    though the file exists.

    Values already present in the real environment always win: an explicit
    `export ANTHROPIC_API_KEY=...` overrides whatever is in the file.
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

    Defaults are plain literals, not os.getenv() calls. Reading the
    environment in field defaults would bind it once at import time, which
    makes the value invisible to tests and dependent on import order -- so the
    environment is read in `from_env()` instead, and `Settings()` on its own is
    a pure, predictable object.
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
    query_timeout_seconds: float = 10.0

    # --- retries -------------------------------------------------------
    # Tenacity owns the retry policy; the SDK's own retries are disabled so
    # attempts are not multiplied (3 x 3 = 9 real calls). See claude.py.
    max_attempts: int = 3
    retry_initial_seconds: float = 1.0
    retry_max_seconds: float = 10.0
    request_timeout_seconds: float = 120.0

    # --- agent ---------------------------------------------------------
    # One repair attempt: enough to fix a mistyped column, not enough to burn
    # budget looping on a fundamentally wrong query.
    max_repair_attempts: int = 1

    @classmethod
    def from_env(cls, **overrides) -> Settings:
        """Build settings from environment variables, then apply overrides.

        Overrides come from CLI flags and always win over the environment.
        """
        env: dict = {}
        for field_name, caster in (
            ("model", str), ("sql_effort", str), ("answer_effort", str),
            ("sql_max_tokens", int), ("answer_max_tokens", int),
            ("db_path", Path), ("max_rows", int),
            ("query_timeout_seconds", float), ("max_repair_attempts", int),
            ("max_attempts", int), ("retry_initial_seconds", float),
            ("retry_max_seconds", float), ("request_timeout_seconds", float),
        ):
            raw = os.getenv(f"NLQ_{field_name.upper()}")
            if raw is None or raw.strip() == "":
                continue
            try:
                env[field_name] = caster(raw.strip())
            except ValueError as exc:
                raise ConfigError(
                    f"NLQ_{field_name.upper()} is not a valid "
                    f"{caster.__name__}: {raw!r}"
                ) from exc
        return cls(**{**env, **{k: v for k, v in overrides.items() if v is not None}})

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


#: Process-wide settings, resolved once from the environment at import.
SETTINGS = Settings.from_env()
