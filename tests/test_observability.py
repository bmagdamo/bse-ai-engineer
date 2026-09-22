"""The audit trail: one record per question, and never the result rows."""

from __future__ import annotations

import json
import logging

from bse_nlq.agent import NLQResult, Outcome
from bse_nlq.claude import TokenUsage
from bse_nlq.observability import AuditLog, AuditRecord, new_request_id


def _result(**overrides) -> NLQResult:
    fields = dict(
        question="How many venues?",
        outcome=Outcome.ANSWERED,
        answer="There are 2 venues.",
        sql="SELECT COUNT(*) FROM venues LIMIT 200",
        columns=("n",),
        rows=((2,),),
        elapsed_seconds=0.25,
        total_seconds=1.5,
        usage=TokenUsage(calls=2, input_tokens=100, output_tokens=20, cache_read_tokens=80),
        request_id="abc123abc123",
        model="claude-opus-5",
        prompt_version="deadbeef1234",
    )
    return NLQResult(**fields | overrides)


def test_request_ids_are_short_and_unique():
    ids = {new_request_id() for _ in range(100)}
    assert len(ids) == 100
    assert all(len(i) == 12 for i in ids)


def test_record_captures_cost_and_provenance():
    record = AuditRecord.of(_result())
    assert record.request_id == "abc123abc123"
    assert record.model == "claude-opus-5"
    assert record.prompt_version == "deadbeef1234"
    assert record.outcome == "answered"
    assert record.api_calls == 2 and record.cache_read_tokens == 80
    assert record.row_count == 1
    assert record.total_seconds == 1.5 and record.query_seconds == 0.25


def test_record_never_carries_result_rows():
    """The audit trail records that a query ran, not what it returned --
    otherwise the log becomes a second, unguarded copy of the data."""
    payload = json.loads(AuditRecord.of(_result()).as_json())
    assert "rows" not in payload and "columns" not in payload
    assert payload["row_count"] == 1, "the count is kept; the data is not"
    assert not any("There are 2 venues" in str(value) for value in payload.values())


def test_every_outcome_is_recorded(caplog):
    log = AuditLog()
    with caplog.at_level(logging.INFO, logger="bse_nlq.audit"):
        for outcome in Outcome:
            log.record(_result(outcome=outcome))
    emitted = [json.loads(r.getMessage())["outcome"] for r in caplog.records]
    assert emitted == [str(o) for o in Outcome]


def test_jsonl_sink_appends_one_line_per_question(tmp_path):
    path = tmp_path / "nested" / "audit.jsonl"
    log = AuditLog(path)
    log.record(_result())
    log.record(_result(outcome=Outcome.DECLINED))

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["outcome"] for line in lines] == ["answered", "declined"]


def test_an_unwritable_sink_warns_once_and_never_fails_the_question(tmp_path, caplog):
    blocked = tmp_path / "file.txt"
    blocked.write_text("not a directory")
    log = AuditLog(blocked / "audit.jsonl")

    with caplog.at_level(logging.WARNING, logger="bse_nlq.audit"):
        for _ in range(3):
            log.record(_result())     # must not raise

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1, "a broken sink warns once per process, not per question"
