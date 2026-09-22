"""A fake ModelClient, so the whole pipeline is testable without network."""

from __future__ import annotations

import json

from bse_nlq.claude import ModelClient, ModelRequest, ModelResponse, TokenUsage
from bse_nlq.config import Settings
from bse_nlq.errors import ModelError

#: Retry policy for unit tests: one attempt, no backoff. A test that sleeps
#: is a test nobody runs.
NO_RETRY = Settings(max_attempts=1, retry_initial_seconds=0.0, retry_max_seconds=0.0)


class FakeModelClient:
    """Returns queued payloads in order and records every request.

    A dict payload is JSON-encoded (a structured-output SqlPlan); a string is
    returned verbatim (a synthesis answer); an exception is raised.
    """

    def __init__(self, *payloads: dict | str | Exception, model: str = ""):
        self._payloads = list(payloads)
        self.model = model
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._payloads:
            raise AssertionError("FakeModelClient ran out of queued payloads")
        payload = self._payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return ModelResponse(
            text=text,
            usage=TokenUsage(calls=1, input_tokens=100, output_tokens=50,
                             cache_read_tokens=10),
            model=self.model,
        )


# The fake must actually satisfy the protocol the agent depends on.
assert isinstance(FakeModelClient(), ModelClient)


def plan(sql: str, **overrides) -> dict:
    """A valid SqlPlan payload, with fields overridable per test."""
    return {
        "is_answerable": True,
        "sql": sql,
        "explanation": "Counts venues.",
        "assumptions": ["Counts every venue on file."],
        "unanswerable_reason": "",
        **overrides,
    }


def declined(reason: str) -> dict:
    return plan("", is_answerable=False, explanation="", assumptions=[],
                unanswerable_reason=reason)


def model_error(message: str) -> ModelError:
    return ModelError(message)
