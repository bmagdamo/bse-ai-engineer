"""Typed errors, each carrying a user-facing message.

Every failure path in the agent maps to one of these so the UIs can render a
clear message instead of a stack trace.
"""

from __future__ import annotations


class NLQError(Exception):
    """Base class. `user_message` is what a non-technical user should see."""

    def __init__(self, user_message: str, detail: str = ""):
        super().__init__(user_message if not detail else f"{user_message} ({detail})")
        self.user_message = user_message
        self.detail = detail


class ConfigError(NLQError):
    """Missing API key, missing database file, bad settings."""


class UnsafeSQLError(NLQError):
    """The generated SQL was not a single read-only SELECT.

    This should never happen given the system prompt; treat it as a prompt bug.
    """


class QueryExecutionError(NLQError):
    """The database rejected the query. Carries the raw SQLite message for repair."""

    def __init__(self, user_message: str, sqlite_message: str, sql: str):
        super().__init__(user_message, sqlite_message)
        self.sqlite_message = sqlite_message
        self.sql = sql


class QueryTimeoutError(NLQError):
    """Query exceeded the configured wall-clock budget and was aborted."""


class ModelError(NLQError):
    """The Claude API call failed (auth, rate limit, overload, network)."""


class TransientModelError(ModelError):
    """A model failure that is worth retrying: rate limit, 5xx, or a network
    blip. Permanent failures (bad key, unknown model, malformed request) use
    the plain ModelError so they fail fast instead of burning the retry budget.

    `retry_after` carries the server's own guidance when it sent a
    `retry-after` header; the retry policy prefers it over its own backoff.
    """

    def __init__(self, user_message: str, detail: str = "",
                 retry_after: float | None = None):
        super().__init__(user_message, detail)
        self.retry_after = retry_after
