"""The structured-output contract between the model and the agent.

Constraints imposed by the structured-outputs API: every field must be
required (so: no defaults) and objects must set additionalProperties: false
(so: extra="forbid").
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SqlPlan(BaseModel):
    """What the model returns for a single natural-language question.

    `is_answerable=False` is a first-class path: it lets the model decline
    structurally instead of inventing a query against columns that do not exist.
    """

    model_config = ConfigDict(extra="forbid")

    is_answerable: bool = Field(
        description="True if the question can be answered with this database alone."
    )
    sql: str = Field(
        description="A single read-only SQLite SELECT. Empty string when not answerable."
    )
    explanation: str = Field(
        description="One plain-English sentence describing what the query computes."
    )
    assumptions: list[str] = Field(
        description="Interpretation choices a user should know about. Empty list if none."
    )
    unanswerable_reason: str = Field(
        description="Why the question cannot be answered. Empty string when answerable."
    )


def sql_plan_json_schema() -> dict:
    """JSON Schema for the structured-outputs `format` parameter."""
    return SqlPlan.model_json_schema()
