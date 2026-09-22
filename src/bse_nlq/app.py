"""Streamlit UI.

Intentionally thin -- it owns no agent logic, only presentation. Run with:

    uv run --extra ui streamlit run src/bse_nlq/app.py
"""

from __future__ import annotations

import streamlit as st

from bse_nlq import formatter
from bse_nlq.agent import NLQAgent
from bse_nlq.config import SETTINGS
from bse_nlq.db import Database
from bse_nlq.errors import NLQError
from bse_nlq.logging_setup import configure as configure_logging
from bse_nlq.prompts import EXAMPLE_QUESTIONS

configure_logging()
st.set_page_config(page_title="BSE Natural Language Query", page_icon="🎟️", layout="wide")


@st.cache_resource
def get_agent() -> NLQAgent:
    SETTINGS.require_database()
    SETTINGS.require_api_key()
    db = Database(SETTINGS.db_path, SETTINGS.max_rows, SETTINGS.query_timeout_seconds)
    return NLQAgent(db, SETTINGS)


st.title("🎟️ BSE Natural Language Query")
st.caption("Ask about events, ticket sales, revenue, venues, and customers in plain English.")

try:
    agent = get_agent()
except NLQError as exc:
    st.error(exc.user_message)
    st.stop()

with st.sidebar:
    st.subheader("Examples")
    for example in EXAMPLE_QUESTIONS[:3]:   # the three from the exercise brief
        if st.button(example, use_container_width=True):
            st.session_state["question"] = example
    st.caption(f"Model: `{SETTINGS.model}`")

question = st.text_input("Your question", key="question",
                         placeholder="e.g. Which concerts sold the most tickets last year?")

if question:
    with st.spinner("Thinking…"):
        result = agent.ask(question)

    if not result.ok:
        st.error(result.answer)
    elif not result.answerable:
        st.warning(result.answer)
    else:
        st.success(result.answer)

    if result.sql:
        st.subheader("Generated SQL")
        st.code(result.sql, language="sql")

    if result.rows:
        st.subheader(f"Results ({result.row_count} rows)")
        st.dataframe(formatter.to_dicts(result), use_container_width=True)
        if result.truncated:
            st.caption(f"Capped at the {result.row_count}-row limit; there may be more.")

    if result.assumptions:
        with st.expander("Assumptions"):
            for item in result.assumptions:
                st.markdown(f"- {item}")

    trace = [f"request `{result.request_id}`"]
    if result.cached:
        trace.append("served from cache")
    if result.usage.calls:
        trace.append(result.usage.summary())
    st.caption(" · ".join(trace))
