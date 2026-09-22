"""Resilience at the model boundary: fallback order and the circuit breaker."""

from __future__ import annotations

import time

import pytest

from bse_nlq.claude import (
    FallbackClient,
    Message,
    ModelClient,
    ModelRequest,
    ModelResponse,
    TokenUsage,
    build_client,
)
from bse_nlq.config import Settings
from bse_nlq.errors import ModelError, TransientModelError

REQUEST = ModelRequest(system="s", messages=(Message("user", "q"),),
                       max_tokens=10, effort="low")

#: Trips after two failures and recovers fast, so the test does not sleep.
QUICK = Settings(breaker_failure_threshold=2, breaker_cooldown_seconds=0.05)


class _Stub:
    """A ModelClient that raises `error` (if any) and counts its calls."""

    def __init__(self, error: Exception | None = None, text: str = "ok",
                 model: str = ""):
        self.error, self.text, self.model, self.calls = error, text, model, 0

    def complete(self, _request: ModelRequest) -> ModelResponse:
        self.calls += 1
        if self.error:
            raise self.error
        return ModelResponse(text=self.text, usage=TokenUsage(calls=1),
                             model=self.model)


def test_fallback_is_itself_a_model_client():
    """The agent must not be able to tell the difference."""
    assert isinstance(FallbackClient(_Stub()), ModelClient)


def test_a_transient_failure_falls_through_to_the_next_model():
    primary, backup = _Stub(TransientModelError("rate limited")), _Stub(text="from backup")
    client = FallbackClient(primary, backup, settings=QUICK)
    assert client.complete(REQUEST).text == "from backup"
    assert primary.calls == 1 and backup.calls == 1


def test_a_fallback_answer_names_the_model_that_served_it():
    """Provenance is read off the response, so the audit trail can tell which
    model answered during an outage instead of naming the configured primary."""
    primary = _Stub(TransientModelError("rate limited"), model="primary-model")
    backup = _Stub(text="from backup", model="backup-model")
    response = FallbackClient(primary, backup, settings=QUICK).complete(REQUEST)
    assert response.model == "backup-model"


def test_a_permanent_failure_does_not_fall_through():
    """A bad key is not an outage: it would fail identically on every model,
    so spending the whole roster on it only delays the same error."""
    primary, backup = _Stub(ModelError("invalid api key")), _Stub()
    with pytest.raises(ModelError, match="invalid api key"):
        FallbackClient(primary, backup, settings=QUICK).complete(REQUEST)
    assert backup.calls == 0


def test_the_circuit_opens_and_stops_calling_a_dead_model():
    primary, backup = _Stub(TransientModelError("down")), _Stub()
    client = FallbackClient(primary, backup, settings=QUICK)
    for _ in range(5):
        client.complete(REQUEST)
    assert primary.calls == QUICK.breaker_failure_threshold, "skipped once open"
    assert backup.calls == 5, "every question is still answered"


def test_the_circuit_closes_again_after_the_cooldown():
    primary, backup = _Stub(TransientModelError("blip")), _Stub()
    client = FallbackClient(primary, backup, settings=QUICK)
    for _ in range(3):
        client.complete(REQUEST)
    assert primary.calls == 2

    time.sleep(QUICK.breaker_cooldown_seconds + 0.01)
    client.complete(REQUEST)
    assert primary.calls == 3, "the next request through is the probe"


def test_a_failed_probe_reopens_the_circuit_immediately():
    """A prolonged outage should cost one probe per cooldown, not a fresh run
    at the threshold -- every readmitted request pays its own retry backoff
    against an endpoint that is still down."""
    primary, backup = _Stub(TransientModelError("still down")), _Stub()
    client = FallbackClient(primary, backup, settings=QUICK)
    for _ in range(3):
        client.complete(REQUEST)
    assert primary.calls == QUICK.breaker_failure_threshold

    time.sleep(QUICK.breaker_cooldown_seconds + 0.01)
    for _ in range(3):
        client.complete(REQUEST)
    assert primary.calls == QUICK.breaker_failure_threshold + 1, (
        "the probe failed, so the circuit should have reopened on that one call"
    )


def test_a_success_resets_the_failure_count():
    flaky, backup = _Stub(TransientModelError("blip")), _Stub()
    client = FallbackClient(flaky, backup, settings=QUICK)
    client.complete(REQUEST)                 # one failure, below the threshold
    flaky.error = None
    client.complete(REQUEST)
    flaky.error = TransientModelError("blip")
    client.complete(REQUEST)
    assert flaky.calls == 3, "the counter restarted, so the circuit stayed closed"


def test_every_model_unavailable_reports_the_last_real_error():
    client = FallbackClient(_Stub(TransientModelError("primary down")),
                            _Stub(TransientModelError("backup down")), settings=QUICK)
    with pytest.raises(TransientModelError, match="backup down"):
        client.complete(REQUEST)


def test_build_client_stays_a_single_client_unless_a_fallback_is_configured(monkeypatch):
    """The default path carries no extra machinery."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")   # the SDK needs one to construct
    assert not isinstance(build_client(Settings()), FallbackClient)
    assert isinstance(build_client(Settings(fallback_model="claude-haiku-4-5")),
                      FallbackClient)
