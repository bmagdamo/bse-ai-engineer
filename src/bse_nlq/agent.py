"""The NLQ agent: question -> SQL -> rows -> plain-English answer.

    plan -> [decline?] -> guard -> execute -> [repair once?] -> synthesize

Two model calls per question. Planning uses structured outputs so "I cannot
answer that" is a parseable state rather than prose we would pattern-match.
Synthesis is a separate, cheaper call that never sees the schema.

The agent depends on the `ModelClient` protocol, not on `anthropic`, so the
whole pipeline is exercisable in tests without network access.

Around that pipeline sit the concerns a long-running deployment needs, each
owned by its own module so `ask()` stays a readable sequence: a correlation id
and audit record per question (`observability`), one end-to-end budget every
stage checks (`deadline`), a repeat-question cache (`cache`), and a schema
fingerprint that rebuilds the prompt when the database changes underneath it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from zoneinfo import ZoneInfo

from bse_nlq import cache, formatter, guards, observability, prompts, schema_context
from bse_nlq.claude import Message, ModelClient, ModelRequest, TokenUsage, build_client
from bse_nlq.config import SETTINGS, Settings
from bse_nlq.db import Database, QueryResult
from bse_nlq.deadline import Deadline
from bse_nlq.errors import ModelError, NLQError, QueryExecutionError
from bse_nlq.models import SqlPlan
from bse_nlq.observability import AuditLog

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
    elapsed_seconds: float = 0.0        # the SQL query alone
    total_seconds: float = 0.0          # the whole question, end to end
    usage: TokenUsage = TokenUsage()
    # Provenance: enough to attribute an answer to the exact model and prompt
    # that produced it, which is what makes an accuracy regression traceable
    # to a release rather than to a guess.
    request_id: str = ""
    model: str = ""
    prompt_version: str = ""
    cached: bool = False

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
    deadline: Deadline
    usage: TokenUsage = TokenUsage()
    attempts: list[Attempt] = field(default_factory=list)
    model: str = ""
    """Whichever model last answered, for provenance. Read back off the
    response rather than assumed from the settings, so a question the fallback
    served is attributed to the fallback."""


class NLQAgent:
    """Stateless across questions -- each ask() is an independent turn.

    The cache, the audit sink and the circuit breakers are per agent, not per
    question: one agent is constructed per process (Streamlit caches it, the
    CLI holds one for the session) and shared across threads.
    """

    def __init__(self, db: Database, settings: Settings = SETTINGS,
                 client: ModelClient | None = None,
                 audit: AuditLog | None = None):
        self.db = db
        self.settings = settings
        self.client = client or build_client(settings)
        self.audit = audit or AuditLog(settings.audit_log_path)
        self.cache: cache.TTLCache[NLQResult] = cache.TTLCache(
            settings.cache_max_entries, settings.cache_ttl_seconds
        )
        # The access policy is resolved once: the baseline rules plus, when
        # columns are restricted, the rule that enforces that.
        self.rules = guards.policy_for(settings.restricted_column_names)
        self._schema_checked_at = time.monotonic()
        self._build_prompt()

    def _build_prompt(self) -> None:
        """(Re)build the system prompt from the live schema.

        Byte-identical between questions, which is what makes the prompt cache
        hit; `prompt_version` is the handle on *which* prompt that was, so a
        cached answer and an audit line both name the revision behind them.
        """
        self.schema_fingerprint = schema_context.fingerprint(self.db)
        self.system_prompt = prompts.build_system_prompt(
            schema_context.build_schema_context(self.db),
            self.settings.restricted_column_names,
        )
        self.prompt_version = cache.key_for(self.system_prompt)[:12]

    def ask(self, question: str) -> NLQResult:
        """Answer one question.

        Never raises for an expected failure: every error path returns an
        NLQResult whose `answer` is safe to show a non-technical user.
        """
        question = (question or "").strip()
        request_id = observability.new_request_id()
        if not question:
            return NLQResult(question, Outcome.FAILED, "Please enter a question.",
                             request_id=request_id, model=self.settings.model)

        result = self._answer(question, request_id)
        # Recorded on every path -- answered, declined, failed, served from
        # cache -- because an audit trail with holes in it answers nothing.
        self.audit.record(result)
        return result

    def _answer(self, question: str, request_id: str) -> NLQResult:
        """The cache-aware body of ask(). One exit, so the correlation id and
        the end-to-end timing are stamped on every outcome exactly once."""
        started = time.monotonic()
        self._refresh_schema()

        key = self._cache_key(question)
        hit = self.cache.get(key)
        if hit is not None:
            log.debug("Cache hit for %r", question)
            # Usage is zeroed, not inherited: serving this cost nothing, and a
            # cost dashboard that sums the field would otherwise bill every
            # cache hit for the calls the original answer made.
            result = replace(hit, cached=True, usage=TokenUsage())
        else:
            deadline = Deadline.after(self.settings.total_deadline_seconds)
            run = _Run(question=question, deadline=deadline,
                       request=self._initial_request(question, deadline))
            try:
                result = self._pipeline(run)
            except NLQError as exc:
                log.warning("Question failed (%s): %s", type(exc).__name__, exc)
                result = self._failure(run, exc.user_message)
            else:
                # Only successes are cached. Replaying a failure would hide a
                # transient outage behind a stale error for the whole TTL.
                self.cache.put(key, result)

        return replace(result, request_id=request_id,
                       total_seconds=time.monotonic() - started)

    def _cache_key(self, question: str) -> str:
        """Everything an answer depends on, so nothing else can be served.

        The business date is in the key because "last month" means a different
        window tomorrow; the schema fingerprint and prompt version are in it
        because a migration or a prompt change invalidates every prior answer;
        the model is in it because two models do not have to agree.
        """
        return cache.key_for(
            question.casefold(),
            business_today(self.settings),
            self.schema_fingerprint,
            self.prompt_version,
            self.settings.model,
        )

    def _refresh_schema(self) -> None:
        """Rebuild the prompt if the database changed under a long-lived process.

        Streamlit caches one agent for the life of the process, so without this
        a migration leaves the prompt describing columns that no longer exist,
        and every question fails until someone restarts it. Rate-limited
        because the check is cheap (PRAGMA only) but not free.
        """
        interval = self.settings.schema_check_interval_seconds
        if interval <= 0 or time.monotonic() - self._schema_checked_at < interval:
            return
        self._schema_checked_at = time.monotonic()
        if schema_context.fingerprint(self.db) == self.schema_fingerprint:
            return
        log.info("Schema changed; rebuilding the prompt and dropping cached answers.")
        self._build_prompt()
        # Entries keyed on the old fingerprint can never be hit again, so this
        # is housekeeping rather than correctness -- but an agent that answers
        # from a schema it no longer describes is worth being explicit about.
        self.cache.clear()

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
            **self._provenance(run),
        )

    def _execute_with_repair(self, run: _Run, plan: SqlPlan
                             ) -> tuple[SqlPlan, QueryResult, str]:
        """Guard and run the query, allowing a bounded number of repairs.

        A guard failure is deliberately *not* repaired: unsafe SQL means the
        prompt is wrong, and retrying would burn tokens on the same bug.
        """
        for remaining in range(self.settings.max_repair_attempts, -1, -1):
            run.deadline.check("running the query")
            sql, limit_added = guards.enforce_limit(
                plan.sql, self.settings.max_rows, self.rules
            )
            try:
                # The query gets whatever is left of the question's budget, not
                # its own full timeout, so a slow query cannot overrun it.
                result = self.db.run_select(
                    sql, run.deadline.clamp(self.db.timeout_seconds)
                )
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

    def _initial_request(self, question: str, deadline: Deadline) -> ModelRequest:
        return ModelRequest(
            system=self.system_prompt,
            messages=(Message("user", prompts.build_question_turn(
                question, business_today(self.settings), self.settings.timezone)),),
            max_tokens=self.settings.sql_max_tokens,
            effort=self.settings.sql_effort,
            response_schema=SqlPlan.model_json_schema(),
            deadline=deadline,
        )

    def _plan(self, run: _Run) -> SqlPlan:
        """Natural language -> SqlPlan, via structured outputs."""
        run.deadline.check("planning the query")
        response = self.client.complete(run.request)
        run.usage += response.usage
        run.model = response.model or run.model
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
        run.deadline.check("writing the answer")
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
            deadline=run.deadline,
        ))
        run.usage += response.usage
        run.model = response.model or run.model
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
            **self._provenance(run),
        )

    def _failure(self, run: _Run, message: str) -> NLQResult:
        return NLQResult(
            question=run.question,
            outcome=Outcome.FAILED,
            answer=message,
            sql=run.attempts[-1].sql if run.attempts else "",
            attempts=tuple(run.attempts),
            usage=run.usage,
            **self._provenance(run),
        )

    def _provenance(self, run: _Run) -> dict[str, str]:
        """Which model and which prompt revision produced this answer.

        The configured model is only the fallback for a run that never reached
        a model at all; otherwise this is the one that actually served it.
        """
        return {"model": run.model or self.settings.model,
                "prompt_version": self.prompt_version}
