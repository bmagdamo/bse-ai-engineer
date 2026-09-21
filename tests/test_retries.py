"""Retry policy: transient failures are retried, permanent ones are not."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import anthropic
import pytest

from bse_nlq.claude import ClaudeClient, Message, ModelRequest
from bse_nlq.config import Settings
from bse_nlq.errors import ModelError, TransientModelError
from tests.fakes import NO_RETRY

FAST = Settings(max_attempts=3, retry_initial_seconds=0.001, retry_max_seconds=0.002)


class _Block:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Reply:
    stop_reason = "end_turn"

    def __init__(self):
        self.content = [_Block("ok")]
        self.usage = SimpleNamespace(input_tokens=1, output_tokens=1,
                                     cache_creation_input_tokens=0,
                                     cache_read_input_tokens=0)


class _FlakySDK:
    """Raises the queued errors, then succeeds. Counts real call attempts."""

    def __init__(self, *errors):
        self._errors = list(errors)
        self.calls = 0
        self.messages = self

    def create(self, **_):
        self.calls += 1
        if self._errors:
            raise self._errors.pop(0)
        return _Reply()


def _status_error(cls, status, headers=None):
    response = SimpleNamespace(status_code=status, headers=headers or {}, request=None)
    return cls("boom", response=response, body=None)


def _request() -> ModelRequest:
    return ModelRequest(system="s", messages=(Message("user", "q"),),
                        max_tokens=10, effort="low")


def _client(sdk) -> ClaudeClient:
    return ClaudeClient("claude-opus-5", sdk, settings=FAST)


# --- transient failures are retried ----------------------------------------

@pytest.mark.parametrize("error", [
    _status_error(anthropic.InternalServerError, 500),
    _status_error(anthropic.RateLimitError, 429),
    anthropic.APIConnectionError(request=None),
])
def test_transient_failures_are_retried_then_succeed(error):
    sdk = _FlakySDK(error)
    assert _client(sdk).complete(_request()).text == "ok"
    assert sdk.calls == 2, "should have retried exactly once before succeeding"


def test_retries_stop_at_max_attempts():
    sdk = _FlakySDK(*[_status_error(anthropic.InternalServerError, 500)] * 5)
    with pytest.raises(TransientModelError):
        _client(sdk).complete(_request())
    assert sdk.calls == FAST.max_attempts, "must not exceed the configured budget"


# --- permanent failures fail fast ------------------------------------------

@pytest.mark.parametrize("error", [
    _status_error(anthropic.AuthenticationError, 401),
    _status_error(anthropic.PermissionDeniedError, 403),
    _status_error(anthropic.NotFoundError, 404),
    _status_error(anthropic.BadRequestError, 400),
])
def test_permanent_failures_are_not_retried(error):
    """Retrying a bad API key just delays the same error behind three sleeps."""
    sdk = _FlakySDK(error)
    with pytest.raises(ModelError) as excinfo:
        _client(sdk).complete(_request())
    assert not isinstance(excinfo.value, TransientModelError)
    assert sdk.calls == 1


# --- policy details ---------------------------------------------------------

def test_sdk_retries_are_disabled_so_attempts_are_not_multiplied():
    """Tenacity owns the policy. If the SDK also retried, 3 attempts would
    become 9 real API calls and the backoff would be unreasonable."""
    client = ClaudeClient("claude-opus-5", settings=FAST)
    assert client._client.max_retries == 0
    assert client._client.timeout == FAST.request_timeout_seconds


def test_retry_after_header_is_honoured():
    error = _status_error(anthropic.RateLimitError, 429, {"retry-after": "7"})
    sdk = _FlakySDK(error)
    with pytest.raises(TransientModelError) as excinfo:
        ClaudeClient("claude-opus-5", sdk,
                     settings=NO_RETRY).complete(_request())
    assert excinfo.value.retry_after == 7.0


def test_malformed_retry_after_is_ignored_not_fatal():
    error = _status_error(anthropic.RateLimitError, 429, {"retry-after": "soon"})
    sdk = _FlakySDK(error)
    with pytest.raises(TransientModelError) as excinfo:
        ClaudeClient("claude-opus-5", sdk,
                     settings=NO_RETRY).complete(_request())
    assert excinfo.value.retry_after is None


def test_retries_are_logged(caplog):
    """A silent retry hides a degrading API from whoever is on call."""
    sdk = _FlakySDK(_status_error(anthropic.InternalServerError, 500))
    with caplog.at_level(logging.WARNING, logger="bse_nlq.claude"):
        _client(sdk).complete(_request())
    assert any("retrying" in record.getMessage().lower() for record in caplog.records)
