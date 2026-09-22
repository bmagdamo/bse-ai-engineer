"""NLQ accuracy eval. Requires ANTHROPIC_API_KEY; skipped without one.

    uv run pytest -m llm -v

Correctness is judged by comparing the agent's result against a hand-written
gold query, NOT by matching SQL text. Comparison is also *position*-
independent: the agent is free to select columns in any order, add extra
columns, or alias differently, so long as the answer it computes is the same.
Grading on column position would reintroduce exactly the brittleness that
result-equivalence scoring exists to avoid.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from bse_nlq.agent import NLQAgent, business_today
from bse_nlq.config import SETTINGS
from bse_nlq.db import Database

pytestmark = pytest.mark.llm

CASES = yaml.safe_load((Path(__file__).parent / "eval_cases.yaml").read_text())
MONEY_TOLERANCE = 0.02


@dataclass(frozen=True, slots=True)
class Cells:
    """A result row split into the parts that carry meaning."""

    labels: tuple[str, ...]     # text cells, e.g. a category or event name
    measures: tuple[float, ...]  # numeric cells, e.g. revenue or a count

    @classmethod
    def of(cls, row) -> Cells:
        labels, measures = [], []
        for value in row:
            if isinstance(value, bool) or value is None:
                continue
            if isinstance(value, (int, float)):
                measures.append(round(float(value), 2))
            else:
                measures.append(float(value)) if _is_number(str(value)) else labels.append(str(value))
        return cls(tuple(labels), tuple(measures))


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _rows(result) -> list[Cells]:
    return [Cells.of(row) for row in result]


# -- comparison strategies ---------------------------------------------------

def _scalar(actual, gold) -> tuple[bool, str]:
    """The single number both queries compute must agree."""
    a, g = _rows(actual), _rows(gold)
    if not a or not g or not a[0].measures or not g[0].measures:
        return False, f"expected one numeric cell (actual={a[:1]}, gold={g[:1]})"
    return (abs(a[0].measures[0] - g[0].measures[0]) <= MONEY_TOLERANCE,
            f"{a[0].measures[0]} != {g[0].measures[0]}")


def _top_label(actual, gold) -> tuple[bool, str]:
    """The top-ranked row must identify the same thing, wherever that name
    sits among the agent's columns."""
    a, g = _rows(actual), _rows(gold)
    if not a or not g or not g[0].labels:
        return False, f"empty result (actual={len(a)}, gold={len(g)})"
    expected = g[0].labels[0]
    return expected in a[0].labels, f"top row {a[0].labels} does not contain {expected!r}"


def _labels_ordered(actual, gold) -> tuple[bool, str]:
    a = [c.labels[0] for c in _rows(actual) if c.labels]
    g = [c.labels[0] for c in _rows(gold) if c.labels]
    return a[: len(g)] == g, f"{a[: len(g)]} != {g}"


def _label_set(actual, gold) -> tuple[bool, str]:
    a = {c.labels[0] for c in _rows(actual) if c.labels}
    g = {c.labels[0] for c in _rows(gold) if c.labels}
    return a == g, f"{sorted(a)} != {sorted(g)}"


COMPARISONS = {
    "scalar": _scalar,
    "top_label": _top_label,
    "labels_ordered": _labels_ordered,
    "label_set": _label_set,
}


@pytest.fixture(scope="module")
def agent():
    if not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    if not SETTINGS.db_path.exists():
        pytest.skip("database not built; run: uv run python data/seed.py")
    with Database(SETTINGS.db_path, SETTINGS.max_rows,
                  SETTINGS.query_timeout_seconds) as db:
        yield NLQAgent(db, SETTINGS)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_eval_case(agent, case):
    result = agent.ask(case["question"])

    if case.get("expect_unanswerable"):
        assert not result.answerable, (
            f"expected the agent to decline, but it produced SQL:\n{result.sql}"
        )
        assert result.answer.strip(), "a declined question still needs an explanation"
        return

    assert result.ok, f"agent failed: {result.answer}"
    assert result.answerable, f"agent wrongly declined: {result.answer}"
    assert result.sql, "no SQL was generated"

    gold_sql = case["gold_sql"].replace("<TODAY>", business_today(SETTINGS))
    gold = agent.db.run_select(gold_sql)
    compare = COMPARISONS[case["compare"]]
    passed, detail = compare(result.rows, gold.rows)
    assert passed, (
        f"[{case['compare']}] {detail}\n\n--- agent SQL ---\n{result.sql}"
        f"\n\n--- gold SQL ---\n{gold_sql}"
    )
    assert result.answer.strip(), "no natural-language answer was produced"


def test_every_case_declares_a_known_comparison():
    """Guards against a typo'd `compare:` silently skipping real checking."""
    for case in CASES:
        if case.get("expect_unanswerable"):
            assert "gold_sql" not in case, f"{case['id']}: declined cases need no gold SQL"
        else:
            assert case.get("compare") in COMPARISONS, f"{case['id']}: bad compare mode"
            assert case.get("gold_sql"), f"{case['id']}: missing gold SQL"
            assert "date('now'" not in case["gold_sql"], (
                f"{case['id']}: gold SQL uses date('now') (UTC) while the agent is "
                "anchored on the business timezone; use <TODAY> instead"
            )
