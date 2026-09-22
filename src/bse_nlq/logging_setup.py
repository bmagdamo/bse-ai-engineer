"""Logging configuration, applied once at an entry point.

A library must not configure logging on import -- that belongs to whoever
runs the code -- so modules only call `logging.getLogger(__name__)` and the
CLI and Streamlit app call `configure()`. Without it the agent's warnings go
nowhere: a repaired query or a rejected unsafe query leaves no trace.
"""

from __future__ import annotations

import logging
import os

from rich.logging import RichHandler

_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

#: Transport loggers pinned to WARNING even under --verbose. At DEBUG these
#: emit `send_request_body` records that echo the outbound request -- the
#: user's question, the full schema prompt, and the result rows -- into the log
#: stream. The SDK vendors its HTTP stack (httpcore2/httpx2 today), so both the
#: vendored and upstream names are listed; child loggers inherit these levels.
_NOISY_LOGGERS = ("httpx", "httpx2", "httpcore", "httpcore2", "anthropic")


def resolve_level(verbose: bool = False) -> int:
    """NLQ_LOG_LEVEL wins; otherwise --verbose means INFO and the default is
    WARNING, so a normal run stays quiet."""
    if verbose:
        return logging.DEBUG
    name = os.getenv("NLQ_LOG_LEVEL", "").strip().upper()
    if name in _LEVELS:
        return getattr(logging, name)
    return logging.WARNING


def configure(verbose: bool = False) -> None:
    logging.basicConfig(
        level=resolve_level(verbose),
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=False)],
        force=True,
    )
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
