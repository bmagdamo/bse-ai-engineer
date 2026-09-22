"""The NLQ agent: question -> SQL -> rows -> plain-English answer.

    plan -> [decline?] -> guard -> execute -> [repair once?] -> synthesize

Two model calls per question. Planning uses structured outputs so "I cannot
answer that" is a parseable state rather than prose we would pattern-match.
Synthesis is a separate, cheaper call that never sees the schema.

The agent depends on the `ModelClient` protocol, not on `anthropic`, so the
whole pipeline is exercisable in tests without network access.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from zoneinfo import ZoneInfo

from bse_nlq import formatter, guards, prompts
from bse_nlq.claude import ClaudeClient, Message, ModelClient, ModelRequest, TokenUsage
from bse_nlq.config import SETTINGS, Settings
from bse_nlq.db import Database, QueryResult
from bse_nlq.errors import ModelError, NLQError, QueryExecutionError
from bse_nlq.models import SqlPlan
from bse_nlq.schema_context import build_schema_context

log = logging.getLogger(__name__)


def business_today(settings: Settings) -> str:
    """Today on the business calendar, as 'YYYY-MM-DD'.

    The single source of truth for "now". date.today() is the host's local
    date and SQLite's date('now') is UTC; anchoring the prompt on one and the
    generated SQL on the other made them disagree for part of every day, and
    on a month boundary that silently shifted a "last month" window by a whole
    month. The eval's gold SQL anchors on this too, so the two cannot drift.
    """
    return datetime.now(ZoneInfo(settings.timezone)).date().isoformat()


class Outcome(StrEnum):
    """How a question ended. Replaces a pair of booleans that could encode
    states which cannot actually happen."""

    ANSWERED = "answered"
    DECLINED = "declined"   # the database cannot answer this; not a failure
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Attempt:
    """One SQL generation attempt, retained so the UI can show the repair."""

    sql: str
    error: str = ""


@dataclass(frozen=True, slots=True)
class NLQResult:
    question: str
    outcome: Outcome
    answer: str
    sql: str = ""
    explanation: str = ""
    assumptions: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()
    rows: tuple[tuple, ...] = ()
    attempts: tuple[Attempt, ...] = ()
    truncated: bool = False
    elapsed_seconds: float = 0.0
    usage: TokenUsage = TokenUsage()

    @property
    def ok(self) -> bool:
        return self.outcome is not Outcome.FAILED

    @property
    def answerable(self) -> bool:
        return self.outcome is not Outcome.DECLINED

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def is_empty(self) -> bool:
        """A valid query that matched nothing -- distinct from a failure."""
        return self.outcome is Outcome.ANSWERED and not self.rows

    @property
    def repaired(self) -> bool:
        return any(attempt.error for attempt in self.attempts)


@dataclass
class _Run:
    """Mutable state for a single ask(), so helpers don't thread tuples."""

    question: str
    request: ModelRequest
    usage: TokenUsage = TokenUsage()
    attempts: list[Attempt] = field(default_factory=list)


class NLQAgent:
    """Stateless across questions -- each ask() is an independent turn."""

    def __init__(self, db: Database, settings: Settings = SETTINGS,
                 client: ModelClient | None = None):
        self.db = db
        self.settings = settings
        self.client = client or ClaudeClient(settings.model, settings=settings)
        # Built once, and byte-identical between questions: that is what makes
        # the prompt cache hit.
        self.system_prompt = prompts.build_system_prompt(build_schema_context(db))

    def ask(self, question: str) -> NLQResult:
        """Answer one question.

        Never raises for an expected failure: every error path returns an
        NLQResult whose `answer` is safe to show a non-technical user.
        """
        question = (question or "").strip()
        if not question:
            return NLQResult(question, Outcome.FAILED, "Please enter a question.")

        run = _Run(question=question, request=self._initial_request(question))
        try:
            return self._pipeline(run)
        except NLQError as exc:
            log.warning("Question failed (%s): %s", type(exc).__name__, exc)
            return self._failure(run, exc.user_message)

    # -- pipeline ---------------------------------------------------------

    def _pipeline(self, run: _Run) -> NLQResult:
        plan = self._plan(run)
        if not plan.is_answerable:
            return self._declined(run, plan)

        plan, result, sql = self._execute_with_repair(run, plan)
        answer = self._synthesize(run, sql, result)

        assumptions = list(plan.assumptions)
        if result.truncated:
            assumptions.append(f"Results were capped at {self.settings.max_rows} rows.")

        return NLQResult(
            question=run.question,
            outcome=Outcome.ANSWERED,
            answer=answer,
            sql=sql,
            explanation=plan.explanation,
            assumptions=tuple(assumptions),
            columns=tuple(result.columns),
            rows=tuple(result.rows),
            attempts=tuple(run.attempts),
            truncated=result.truncated,
            elapsed_seconds=result.elapsed_seconds,
            usage=run.usage,
        )

    def _execute_with_repair(self, run: _Run, plan: SqlPlan
                             ) -> tuple[SqlPlan, QueryResult, str]:
        """Guard and run the query, allowing a bounded number of repairs.

        A guard failure is deliberately *not* repaired: unsafe SQL means the
        prompt is wrong, and retrying would burn tokens on the same bug.
        """
        for remaining in range(self.settings.max_repair_attempts, -1, -1):
            sql, limit_added = guards.enforce_limit(plan.sql, self.settings.max_rows)
            try:
                result = self.db.run_select(sql)
            except QueryExecutionError as exc:
                run.attempts.append(Attempt(sql=sql, error=exc.sqlite_message))
                if remaining == 0:
                    raise ModelError(
                        "I generated a query but the database rejected it, and my "
                        f"follow-up correction failed too. SQLite said: {exc.sqlite_message}"
                    ) from exc
                log.warning("Repairing SQL after error: %s", exc.sqlite_message)
                run.request = run.request.with_messages(
                    Message("assistant", plan.model_dump_json()),
                    Message("user", prompts.build_repair_turn(sql, exc.sqlite_message)),
                )
                plan = self._plan(run)
                if not plan.is_answerable:
                    raise ModelError(
                        plan.unanswerable_reason or "The query could not be repaired."
                    ) from exc
                continue

            run.attempts.append(Attempt(sql=sql))
            result.truncated = result.truncated or self._capped(result, limit_added)
            return plan, result, sql

        raise AssertionError("unreachable: Settings rejects a negative repair count")

    def _capped(self, result: QueryResult, limit_added: bool) -> bool:
        """Did *our* row cap bind this result?

        Database.run_select detects truncation by fetching one row past its
        cap, but guards.enforce_limit has already put LIMIT max_rows into the
        SQL, so that row can never come back and the flag stays False however
        many rows matched. Reconstruct it here.

        Only a limit *we* injected counts: a model-authored "LIMIT 10" for a
        top-10 question is a complete answer, not a truncated one.
        """
        return limit_added and len(result.rows) >= self.settings.max_rows

    # -- model calls ------------------------------------------------------

    def _initial_request(self, question: str) -> ModelRequest:
        return ModelRequest(
            system=self.system_prompt,
            messages=(Message("user", prompts.build_question_turn(
                question, business_today(self.settings), self.settings.timezone)),),
            max_tokens=self.settings.sql_max_tokens,
            effort=self.settings.sql_effort,
            response_schema=SqlPlan.model_json_schema(),
        )

    def _plan(self, run: _Run) -> SqlPlan:
        """Natural language -> SqlPlan, via structured outputs."""
        response = self.client.complete(run.request)
        run.usage += response.usage
        try:
            return SqlPlan.model_validate_json(response.text)
        except ValueError as exc:
            # Structured outputs make this near-impossible, but a malformed
            # payload should be a clean message rather than a traceback.
            raise ModelError(
                "The model returned a response I could not read. Please try rephrasing.",
                f"{exc}: {response.text[:200]}",
            ) from exc

    def _synthesize(self, run: _Run, sql: str, result: QueryResult) -> str:
        """Rows -> plain-English answer. Cheap call; no schema needed."""
        response = self.client.complete(ModelRequest(
            system=prompts.ANSWER_SYSTEM_PROMPT,
            messages=(Message("user", prompts.build_answer_turn(
                question=run.question,
                sql=sql,
                rendered_rows=formatter.to_markdown(result),
                row_count=len(result.rows),
                truncated=result.truncated,
            )),),
            max_tokens=self.settings.answer_max_tokens,
            effort=self.settings.answer_effort,
        ))
        run.usage += response.usage
        return response.text.strip()

    # -- result builders --------------------------------------------------

    def _scope(self) -> str:
        """What this database can answer -- appended when a question is declined."""
        tables = ", ".join(table.name for table in self.db.tables())
        return (
            "This database covers BSE ticketing: events, venues, teams, seating "
            f"sections, customers, orders and tickets ({tables})."
        )

    def _declined(self, run: _Run, plan: SqlPlan) -> NLQResult:
        reason = plan.unanswerable_reason or "This question cannot be answered from this database."
        log.info("Declined %r: %s", run.question, reason)
        return NLQResult(
            question=run.question,
            outcome=Outcome.DECLINED,
            answer=f"{reason}\n\n{self._scope()}",
            usage=run.usage,
        )

    def _failure(self, run: _Run, message: str) -> NLQResult:
        return NLQResult(
            question=run.question,
            outcome=Outcome.FAILED,
            answer=message,
            sql=run.attempts[-1].sql if run.attempts else "",
            attempts=tuple(run.attempts),
            usage=run.usage,
        )
