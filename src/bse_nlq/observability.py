"""One structured record per question: the request log and the audit trail.

The rich console logging is for a human watching a REPL. This is the other
audience -- a log aggregator that has to answer "what is p95 latency", "what
does a question cost", "how often do we repair, decline, or serve from cache"
without parsing prose, and a compliance reviewer who has to answer "who ran
what SQL against the customer table, and when".

Both are the same record, emitted to two sinks: a JSON line on the
`bse_nlq.audit` logger, and an append-only JSONL file when one is configured.

What is deliberately absent is result rows. The audit trail needs to record
that a query read `customers`, not what it read back; writing rows to disk
would turn the log into a second, unguarded copy of the data.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:                      # imported for typing only: agent.py
    from bse_nlq.agent import NLQResult  # imports this module, not the reverse

log = logging.getLogger("bse_nlq.audit")


def new_request_id() -> str:
    """A short correlation id, stamped on the result and every log line."""
    return uuid.uuid4().hex[:12]


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """The flat, serialisable shape of one answered question."""

    request_id: str
    timestamp: str
    question: str
    outcome: str
    model: str
    prompt_version: str
    cached: bool
    repaired: bool
    sql: str
    row_count: int
    truncated: bool
    total_seconds: float
    query_seconds: float
    api_calls: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int

    @classmethod
    def of(cls, result: NLQResult) -> AuditRecord:
        return cls(
            request_id=result.request_id,
            timestamp=datetime.now(UTC).isoformat(timespec="milliseconds"),
            question=result.question,
            outcome=str(result.outcome),
            model=result.model,
            prompt_version=result.prompt_version,
            cached=result.cached,
            repaired=result.repaired,
            sql=result.sql,
            row_count=result.row_count,
            truncated=result.truncated,
            total_seconds=round(result.total_seconds, 3),
            query_seconds=round(result.elapsed_seconds, 3),
            api_calls=result.usage.calls,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            cache_read_tokens=result.usage.cache_read_tokens,
        )

    def as_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


class AuditLog:
    """Emits each record to the log, and to a JSONL file when configured.

    Writing is best-effort by design: an unwritable audit path is worth a
    warning, but losing the answer to a business question because the disk is
    full is a worse outcome than losing the audit line. Regulated deployments
    that need the opposite should fail here -- which is why the decision sits
    in one method rather than being spread across call sites.
    """

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._warned = False

    def record(self, result: NLQResult) -> AuditRecord:
        record = AuditRecord.of(result)
        log.info("%s", record.as_json())
        if self.path is not None:
            self._append(record)
        return record

    def _append(self, record: AuditRecord) -> None:
        assert self.path is not None
        try:
            # One lock and one open per record: an audit sink writes once per
            # question, so a held file handle would buy nothing and lose
            # records whenever the process is killed rather than shut down.
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(record.as_json() + "\n")
        except OSError as exc:
            if not self._warned:      # once per process, not once per question
                self._warned = True
                log.warning("Audit log at %s is not writable: %s", self.path, exc)
