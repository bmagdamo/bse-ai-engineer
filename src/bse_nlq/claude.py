"""The boundary between the agent and the Claude API.

Everything provider-specific lives here: request shaping, prompt-cache
placement, error translation, usage accounting. The agent depends on the
`ModelClient` protocol rather than on `anthropic`, which keeps the pipeline
testable without network access and confines an SDK change to this file.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

import anthropic
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    stop_any,
    wait_exponential_jitter,
)

from bse_nlq.config import SETTINGS, Settings
from bse_nlq.deadline import Deadline
from bse_nlq.errors import ModelError, TransientModelError

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Message:
    """One conversation turn."""

    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Immutable token tally. Accumulate with `+`."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            calls=self.calls + other.calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
        )

    @property
    def cache_hit(self) -> bool:
        return self.cache_read_tokens > 0

    def summary(self) -> str:
        return (
            f"{self.calls} API calls · in {self.input_tokens:,} · "
            f"out {self.output_tokens:,} · cached {self.cache_read_tokens:,}"
        )


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """A provider-agnostic completion request."""

    system: str
    messages: tuple[Message, ...]
    max_tokens: int
    effort: str
    response_schema: dict | None = None
    """When set, the model is constrained to emit JSON matching this schema."""
    deadline: Deadline | None = None
    """The end-to-end budget for the question this call belongs to. The client
    lowers its own timeout to fit, and stops retrying once it is spent."""

    def with_messages(self, *messages: Message) -> ModelRequest:
        return replace(self, messages=self.messages + messages)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    text: str
    usage: TokenUsage


@runtime_checkable
class ModelClient(Protocol):
    """What the agent needs from a language model: `ClaudeClient` in
    production, a fake in the tests."""

    def complete(self, request: ModelRequest) -> ModelResponse: ...


def _wait_policy(settings: Settings):
    """Exponential backoff with jitter, deferring to a server-supplied
    `retry-after` -- waiting less than the service asked for is how you get
    rate-limited again immediately.

    That figure is honoured in full rather than clamped to retry_max_seconds,
    which bounds our *own* guess and is routinely shorter than a real
    rate-limit window. retry_after_max_seconds is the separate, larger ceiling
    that stops a pathological header from hanging the CLI.
    """
    backoff = wait_exponential_jitter(
        initial=settings.retry_initial_seconds, max=settings.retry_max_seconds
    )

    def wait(state: RetryCallState) -> float:
        exc = state.outcome.exception() if state.outcome else None
        retry_after = getattr(exc, "retry_after", None)
        if retry_after is not None:
            return min(float(retry_after), settings.retry_after_max_seconds)
        return backoff(state)

    return wait


def _stop_policy(settings: Settings, deadline: Deadline | None):
    """Stop on the attempt budget, or as soon as the question's budget is gone.

    Without the second condition the attempt budget wins: a request whose
    deadline passed during the first backoff still sleeps and retries twice
    more before anyone notices there is no time left to use the answer.
    """
    attempts = stop_after_attempt(settings.max_attempts)
    if deadline is None:
        return attempts
    return stop_any(attempts, lambda _state: deadline.expired)


def _retry_logger(settings: Settings):
    """A `before_sleep` hook that reports the attempt budget it was built with.

    The budget is passed in rather than read back off `state.retry_object.stop`:
    with a deadline that policy is a `stop_any`, which carries no
    max_attempt_number, and an AttributeError raised inside before_sleep
    escapes the retry loop as a failure nothing above it is catching.
    """
    def log_retry(state: RetryCallState) -> None:
        exc = state.outcome.exception() if state.outcome else None
        log.warning(
            "Claude API call failed (attempt %d/%d): %s -- retrying in %.1fs",
            state.attempt_number, settings.max_attempts, exc, state.idle_for,
        )

    return log_retry


class ClaudeClient:
    """`ModelClient` backed by the Anthropic API.

    Tenacity owns retries and the SDK's own loop is disabled
    (`max_retries=0`). Both enabled multiplies attempts -- 3 over 3 is 9 real
    calls -- and makes the effective backoff impossible to reason about.
    """

    def __init__(self, model: str, client: anthropic.Anthropic | None = None,
                 settings: Settings | None = None):
        # Resolved here, not bound to the module singleton as a default arg at
        # import: a caller with its own Settings (CLI flags, tests) would
        # otherwise have its retry policy and timeout silently ignored.
        settings = settings or SETTINGS
        self.model = model
        self.settings = settings
        self._client = client or anthropic.Anthropic(
            max_retries=0,                                  # tenacity owns retries
            timeout=settings.request_timeout_seconds,       # never hang forever
        )                                                   # lowered per call by
        #                                                     the deadline, below


    def complete(self, request: ModelRequest) -> ModelResponse:
        """Send one request, retrying only transient failures."""
        retrying = Retrying(
            retry=retry_if_exception_type(TransientModelError),
            stop=_stop_policy(self.settings, request.deadline),
            wait=_wait_policy(self.settings),
            before_sleep=_retry_logger(self.settings),
            reraise=True,
        )
        try:
            return retrying(self._complete_once, request)
        except TransientModelError:
            # Retries that ran out of time report the budget, not the last
            # blip: "the API was busy" is the cause, but "this took too long"
            # is the thing the user can act on.
            if request.deadline is not None and request.deadline.expired:
                request.deadline.check("the model call")
            raise

    def _complete_once(self, request: ModelRequest) -> ModelResponse:
        timeout = self.settings.request_timeout_seconds
        if request.deadline is not None:
            request.deadline.check("the model call")
            timeout = request.deadline.clamp(timeout)

        output_config: dict = {"effort": request.effort}
        if request.response_schema is not None:
            output_config["format"] = {
                "type": "json_schema",
                "schema": request.response_schema,
            }

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=request.max_tokens,
                # The cache breakpoint sits on the system block because it is
                # byte-stable across questions; see prompts.py.
                system=[{
                    "type": "text",
                    "text": request.system,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[m.as_dict() for m in request.messages],
                output_config=output_config,
                # Per call, not per client: the ceiling is whichever is
                # nearer, this call's own limit or what the question has left.
                timeout=timeout,
            )
        except anthropic.APIError as exc:
            raise self._translate(exc) from exc

        self._check_stop_reason(response.stop_reason)
        return ModelResponse(text=self._text_of(response), usage=self._usage_of(response))

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _text_of(response) -> str:
        return "".join(block.text for block in response.content if block.type == "text")

    @staticmethod
    def _usage_of(response) -> TokenUsage:
        usage = response.usage
        return TokenUsage(
            calls=1,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        )

    @staticmethod
    def _check_stop_reason(stop_reason: str | None) -> None:
        if stop_reason == "refusal":
            raise ModelError("The model declined to answer this request.")
        if stop_reason == "max_tokens":
            raise ModelError(
                "The model's response was cut off. Try a simpler question, or raise "
                "NLQ_SQL_MAX_TOKENS."
            )

    def _translate(self, exc: anthropic.APIError) -> ModelError:
        """Map the SDK exception hierarchy onto one actionable message.

        Transient failures become TransientModelError so the retry policy
        picks them up; everything else fails fast -- retrying a bad API key
        just delays the same error behind three backoffs.
        """
        if isinstance(exc, anthropic.AuthenticationError):
            return ModelError(
                "Claude API authentication failed. Check that ANTHROPIC_API_KEY is valid."
            )
        if isinstance(exc, anthropic.PermissionDeniedError):
            return ModelError(
                f"Your API key does not have access to {self.model}. "
                "Set NLQ_MODEL to a model you can use."
            )
        if isinstance(exc, anthropic.NotFoundError):
            return ModelError(f"Model '{self.model}' was not found.")

        if isinstance(exc, anthropic.RateLimitError):
            retry_after = self._retry_after(exc)
            hint = f"{retry_after:.0f}" if retry_after is not None else "a few"
            return TransientModelError(
                f"Rate limited by the Claude API. Try again in {hint} seconds.",
                retry_after=retry_after,
            )
        if isinstance(exc, anthropic.APIConnectionError):
            return TransientModelError(
                "Could not reach the Claude API. Check your network connection."
            )
        if isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500:
            return TransientModelError(
                "The Claude API is temporarily unavailable. Try again shortly."
            )
        return ModelError("The Claude API rejected the request.", str(exc))

    @staticmethod
    def _retry_after(exc: anthropic.APIStatusError) -> float | None:
        response = getattr(exc, "response", None)
        raw = response.headers.get("retry-after") if response is not None else None
        try:
            return float(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None


class _Breaker:
    """Trips a client out of rotation after consecutive transient failures.

    Retrying is the right response to one bad call and the wrong response to a
    dead endpoint: every request then pays the full backoff budget before
    failing the same way. The breaker turns the second case into an immediate
    skip, and closes itself again after a cooldown rather than needing a
    probe -- the next request through is the probe.
    """

    def __init__(self, threshold: int, cooldown_seconds: float):
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self._failures = 0
        self._opened_at = 0.0
        self._lock = threading.Lock()

    @property
    def open(self) -> bool:
        """True while the client is out of rotation."""
        with self._lock:
            if self._failures < self.threshold:
                return False
            if time.monotonic() - self._opened_at >= self.cooldown_seconds:
                self._failures = 0      # cooled down: let the next call probe
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures == self.threshold:
                self._opened_at = time.monotonic()
                log.warning("Model circuit opened for %.0fs after %d failures",
                            self.cooldown_seconds, self._failures)


class FallbackClient:
    """Tries each client in order, skipping any whose circuit is open.

    Composes `ModelClient`s rather than extending one, so the fallback is
    itself a `ModelClient` and the agent cannot tell the difference. Adding a
    third model, or a Bedrock-backed one, is another constructor argument.

    Only transient failures fall through. A bad API key or an unknown model is
    not an outage and would fail identically on every client, so it propagates
    at once instead of being retried across the whole roster.
    """

    def __init__(self, *clients: ModelClient, settings: Settings = SETTINGS):
        if not clients:
            raise ValueError("FallbackClient needs at least one client")
        self._clients = [
            (client, _Breaker(settings.breaker_failure_threshold,
                              settings.breaker_cooldown_seconds))
            for client in clients
        ]

    def complete(self, request: ModelRequest) -> ModelResponse:
        last: TransientModelError | None = None
        for client, breaker in self._clients:
            if breaker.open:
                continue
            try:
                response = client.complete(request)
            except TransientModelError as exc:
                breaker.record_failure()
                last = exc
                log.warning("Model unavailable (%s); trying the next one", exc)
                continue
            breaker.record_success()
            return response

        if last is not None:
            raise last
        raise ModelError(
            "Every configured model is temporarily unavailable. Try again shortly."
        )


def build_client(settings: Settings = SETTINGS) -> ModelClient:
    """The client the agent should use for these settings.

    A single `ClaudeClient` unless a fallback model is configured, so the
    default path stays exactly as simple as it was and resilience is one
    environment variable away.
    """
    primary = ClaudeClient(settings.model, settings=settings)
    if not settings.fallback_model.strip():
        return primary
    log.info("Fallback model configured: %s", settings.fallback_model)
    return FallbackClient(
        primary,
        ClaudeClient(settings.fallback_model, settings=settings),
        settings=settings,
    )
