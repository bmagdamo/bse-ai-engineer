"""The end-to-end budget: it must bound the sum of the stages, not each one."""

from __future__ import annotations

import time

import pytest

from bse_nlq.deadline import Deadline
from bse_nlq.errors import DeadlineExceededError


def test_remaining_shrinks_and_floors_at_zero():
    deadline = Deadline.after(0.05)
    assert 0 < deadline.remaining <= 0.05
    time.sleep(0.06)
    assert deadline.remaining == 0.0, "a negative timeout is never a usable one"
    assert deadline.expired


def test_clamp_lowers_a_stage_timeout_to_what_is_left():
    deadline = Deadline.after(2.0)
    assert deadline.clamp(20.0) <= 2.0
    assert deadline.clamp(0.5) == 0.5, "a shorter stage timeout still wins"


def test_check_names_the_stage_that_could_not_start():
    deadline = Deadline.after(0.0)
    with pytest.raises(DeadlineExceededError) as exc:
        deadline.check("running the query")
    assert "running the query" in exc.value.detail
    assert "too long" in exc.value.user_message, "the user message must be plain English"


def test_check_is_silent_while_budget_remains():
    assert Deadline.after(10).check("planning") is None
