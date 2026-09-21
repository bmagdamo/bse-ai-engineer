"""The model boundary: usage accounting, request shaping, error translation."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import anthropic
import pytest

from bse_nlq.claude import ClaudeClient, Message, ModelRequest, TokenUsage
from bse_nlq.errors import ModelError
from tests.fakes import NO_RETRY

# --- TokenUsage -------------------------------------------------------------

def test_usage_adds_immutably():
    a = TokenUsage(calls=1, input_tokens=10, cache_read_tokens=5)
    b = TokenUsage(calls=1, input_tokens=20)
    total = a + b
    assert total == TokenUsage(calls=2, input_tokens=30, cache_read_tokens=5)
    assert a.calls == 1, "operands must not be mutated"


def test_usage_reports_cache_hits():
    assert not TokenUsage(calls=1).cache_hit
    assert TokenUsage(calls=1, cache_read_tokens=1).cache_hit
    assert "API calls" in TokenUsage(calls=2).summary()


# --- request shaping --------------------------------------------------------

@dataclass
class _Block:
    text: str
    type: str = "text"


@dataclass
class _Usage:
    input_tokens: int = 7
    output_tokens: int = 3
    cache_creation_input_tokens: int = 2
    cache_read_input_tokens: int = 11


@dataclass
class _Reply:
    content: list
    stop_reason: str = "end_turn"
    usage: _Usage = None

    def __post_init__(self):
        self.usage = self.usage or _Usage()


class _RecordingSDK:
    def __init__(self, reply=None, error=None):
        self._reply, self._error = reply, error
        self.kwargs = None
        self.messages = self

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self._error:
            raise self._error
        return self._reply


def build(schema=None) -> ModelRequest:
    return ModelRequest(system="SYSTEM", messages=(Message("user", "hi"),),
                        max_tokens=100, effort="medium", response_schema=schema)


def test_request_carries_cache_breakpoint_and_effort():
    sdk = _RecordingSDK(_Reply([_Block("ok")]))
    ClaudeClient("claude-opus-5", sdk, settings=NO_RETRY).complete(build())

    assert sdk.kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert sdk.kwargs["system"][0]["text"] == "SYSTEM"
    assert sdk.kwargs["output_config"]["effort"] == "medium"
    assert "format" not in sdk.kwargs["output_config"]
    assert sdk.kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_response_schema_becomes_structured_output():
    sdk = _RecordingSDK(_Reply([_Block("{}")]))
    ClaudeClient("claude-opus-5", sdk, settings=NO_RETRY).complete(build(schema={"type": "object"}))
    fmt = sdk.kwargs["output_config"]["format"]
    assert fmt == {"type": "json_schema", "schema": {"type": "object"}}


def test_usage_is_extracted_from_the_reply():
    sdk = _RecordingSDK(_Reply([_Block("ok")]))
    usage = ClaudeClient("claude-opus-5", sdk, settings=NO_RETRY).complete(build()).usage
    assert usage == TokenUsage(calls=1, input_tokens=7, output_tokens=3,
                               cache_write_tokens=2, cache_read_tokens=11)


def test_only_text_blocks_are_concatenated():
    reply = _Reply([_Block("thinking...", type="thinking"), _Block("A"), _Block("B")])
    text = ClaudeClient("claude-opus-5", _RecordingSDK(reply), settings=NO_RETRY).complete(build()).text
    assert text == "AB"


def test_with_messages_appends_without_mutating():
    original = build()
    extended = original.with_messages(Message("assistant", "x"))
    assert len(original.messages) == 1
    assert len(extended.messages) == 2


# --- error translation ------------------------------------------------------

def _http_error(cls, status: int, headers=None):
    """Build an SDK exception without importing the SDK's HTTP library.

    anthropic vendors its transport (httpx2 today), so importing that name in
    a test would couple us to a transitive dependency that can be renamed. The
    exception classes only read status_code/headers, so a duck-typed response
    is both sufficient and more durable.
    """
    response = SimpleNamespace(status_code=status, headers=headers or {}, request=None)
    return cls("boom", response=response, body=None)


@pytest.mark.parametrize("error,expected", [
    (_http_error(anthropic.AuthenticationError, 401), "authentication failed"),
    (_http_error(anthropic.PermissionDeniedError, 403), "does not have access"),
    (_http_error(anthropic.NotFoundError, 404), "was not found"),
    (_http_error(anthropic.InternalServerError, 500), "temporarily unavailable"),
    (_http_error(anthropic.BadRequestError, 400), "rejected the request"),
])
def test_api_errors_become_actionable_messages(error, expected):
    client = ClaudeClient("claude-opus-5", _RecordingSDK(error=error), settings=NO_RETRY)
    with pytest.raises(ModelError) as excinfo:
        client.complete(build())
    assert expected in excinfo.value.user_message.lower()


def test_rate_limit_surfaces_the_retry_after_header():
    error = _http_error(anthropic.RateLimitError, 429, {"retry-after": "30"})
    client = ClaudeClient("claude-opus-5", _RecordingSDK(error=error), settings=NO_RETRY)
    with pytest.raises(ModelError) as excinfo:
        client.complete(build())
    assert "30 seconds" in excinfo.value.user_message


def test_connection_errors_are_translated():
    client = ClaudeClient("claude-opus-5",
                          _RecordingSDK(error=anthropic.APIConnectionError(request=None)),
                          settings=NO_RETRY)
    with pytest.raises(ModelError, match="Could not reach"):
        client.complete(build())


@pytest.mark.parametrize("stop_reason,expected", [
    ("refusal", "declined"),
    ("max_tokens", "cut off"),
])
def test_abnormal_stop_reasons_raise(stop_reason, expected):
    reply = _Reply([_Block("partial")], stop_reason=stop_reason)
    client = ClaudeClient("claude-opus-5", _RecordingSDK(reply), settings=NO_RETRY)
    with pytest.raises(ModelError) as excinfo:
        client.complete(build())
    assert expected in excinfo.value.user_message.lower()
