"""Logging is configured at entry points only, and stays quiet by default."""

from __future__ import annotations

import logging

import pytest

from bse_nlq.logging_setup import _NOISY_LOGGERS, configure, resolve_level


def test_default_is_quiet():
    assert resolve_level() == logging.WARNING


def test_verbose_turns_on_debug():
    assert resolve_level(verbose=True) == logging.DEBUG


def test_env_var_sets_the_level(monkeypatch):
    monkeypatch.setenv("NLQ_LOG_LEVEL", "info")
    assert resolve_level() == logging.INFO


def test_unknown_level_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("NLQ_LOG_LEVEL", "LOUD")
    assert resolve_level() == logging.WARNING


def test_verbose_beats_the_env_var(monkeypatch):
    monkeypatch.setenv("NLQ_LOG_LEVEL", "ERROR")
    assert resolve_level(verbose=True) == logging.DEBUG


def test_sdk_http_logging_is_suppressed():
    """The SDK's HTTP debug logs can echo request bodies, which would put the
    user's question and the schema into the log stream."""
    configure(verbose=True)
    for noisy in _NOISY_LOGGERS:
        assert logging.getLogger(noisy).level == logging.WARNING
    # httpcore2 is the vendored transport that actually emits request bodies.
    assert "httpcore2" in _NOISY_LOGGERS


def test_importing_a_module_does_not_configure_logging():
    """Libraries must not hijack the root logger on import; only configure()
    may touch it."""
    logging.getLogger().handlers.clear()
    import importlib

    import bse_nlq.agent
    importlib.reload(bse_nlq.agent)
    assert logging.getLogger().handlers == []


@pytest.fixture(autouse=True)
def _restore_logging():
    yield
    configure(verbose=False)
