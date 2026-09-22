"""A single wall-clock budget for one question.

Every stage already had its own timeout -- the API call, the SQL query -- but
nothing bounded their *sum*. Three retries against a slow API, across a plan
call, a repair and a synthesis call, can stack into minutes while a user waits
on a prompt that promised an interactive answer.

`Deadline` is the one budget every stage checks itself against: it refuses to
start work that cannot finish, and lowers each stage's own timeout to what is
actually left.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from bse_nlq.errors import DeadlineExceededError


@dataclass(frozen=True, slots=True)
class Deadline:
    """An instant on the monotonic clock after which work should stop.

    Monotonic, not wall clock: a clock adjustment mid-question must not
    lengthen or shorten the budget.
    """

    expires_at: float

    @classmethod
    def after(cls, seconds: float) -> Deadline:
        return cls(time.monotonic() + seconds)

    @property
    def remaining(self) -> float:
        """Seconds left, floored at zero so it is always a usable timeout."""
        return max(0.0, self.expires_at - time.monotonic())

    @property
    def expired(self) -> bool:
        return self.remaining <= 0.0

    def clamp(self, seconds: float) -> float:
        """A per-stage timeout, lowered to fit what is left of the budget."""
        return min(seconds, self.remaining)

    def check(self, stage: str) -> None:
        """Raise before `stage` starts if the budget is already spent.

        Checking up front is what keeps the failure legible: without it the
        stage runs, times out on its own clock, and reports whichever error
        that stage happens to produce.
        """
        if self.expired:
            raise DeadlineExceededError(
                "This question took too long overall and was stopped. Try asking "
                "about a narrower time range.",
                f"deadline exceeded before {stage}",
            )
