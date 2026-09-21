"""Settings must read the environment at call time, not at import time."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from bse_nlq.config import REPO_ROOT, Settings, _load_env
from bse_nlq.errors import ConfigError


def test_defaults_are_literals_not_environment_reads(monkeypatch):
    """Settings() must be pure: reading env in field defaults would bind the
    value once at import and make it invisible to tests."""
    monkeypatch.setenv("NLQ_MODEL", "should-be-ignored")
    assert Settings().model == "claude-opus-5"


def test_from_env_reads_the_environment(monkeypatch):
    monkeypatch.setenv("NLQ_MODEL", "claude-haiku-4-5")
    monkeypatch.setenv("NLQ_MAX_ROWS", "7")
    settings = Settings.from_env()
    assert settings.model == "claude-haiku-4-5"
    assert settings.max_rows == 7


def test_explicit_overrides_beat_the_environment(monkeypatch):
    monkeypatch.setenv("NLQ_MODEL", "from-env")
    assert Settings.from_env(model="from-flag").model == "from-flag"


def test_none_overrides_are_ignored(monkeypatch):
    """Unset CLI flags arrive as None and must not clobber the environment."""
    monkeypatch.setenv("NLQ_MODEL", "from-env")
    assert Settings.from_env(model=None).model == "from-env"


def test_blank_environment_values_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("NLQ_MAX_ROWS", "   ")
    assert Settings.from_env().max_rows == Settings().max_rows


def test_malformed_environment_values_fail_loudly(monkeypatch):
    monkeypatch.setenv("NLQ_MAX_ROWS", "not-a-number")
    with pytest.raises(ConfigError, match="NLQ_MAX_ROWS"):
        Settings.from_env()


def test_settings_are_immutable():
    with pytest.raises(FrozenInstanceError):
        Settings().model = "nope"


def test_missing_api_key_names_the_dotenv_file(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfigError) as excinfo:
        Settings().require_api_key()
    assert ".env" in excinfo.value.user_message


def test_missing_database_says_how_to_build_it(tmp_path):
    with pytest.raises(ConfigError, match="seed.py"):
        Settings(db_path=tmp_path / "absent.db").require_database()


def test_dotenv_is_located_from_the_repo_root(tmp_path, monkeypatch):
    """`nlq` must find .env even when invoked from another directory -- a bare
    load_dotenv() walks up from the cwd and would miss it entirely."""
    monkeypatch.chdir(tmp_path)
    found = _load_env()
    expected = REPO_ROOT / ".env"
    assert found == expected if expected.is_file() else found is None


def test_exported_env_var_beats_the_dotenv_file(monkeypatch):
    import os
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-shell")
    _load_env()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-shell"
