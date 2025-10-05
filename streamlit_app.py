"""Minimal Streamlit front-end for exploring the customer query agent."""
from typing import Any, Dict, List

import pandas as pd

import streamlit as st

from customer_query_agent import BENCHMARK_CASES, GeminiClient, build_agent, load_env_from_file


def ensure_agent() -> Any:
    """Load the query agent into session state if it is not already cached."""
    if "agent" not in st.session_state:
        load_env_from_file()
        st.session_state.agent = build_agent()
    return st.session_state.agent


def render_retrieval(table: List[Dict[str, Any]]) -> None:
    if not table:
        st.info("No retrieval records for this question yet.")
        return
    rows = [
        {
            "record_id": record.get("record_id"),
            "similarity": record.get("similarity"),
            "text": record.get("text", "")[:160],
        }
        for record in table
    ]
    st.dataframe(rows)


def render_supporting_facts(facts: List[Any]) -> None:
    if not facts:
        return
    st.markdown("### Supporting Facts")
    for idx, fact in enumerate(facts, start=1):
        st.write(f"Fact {idx}")
        st.json(fact, expanded=False)


def main() -> None:
    st.set_page_config(page_title="Customer Query Agent", layout="wide")
    st.title("Customer Query Explorer")

    agent = ensure_agent()
    if agent.llm_client is None:
        refreshed_client = GeminiClient.from_env()
        if refreshed_client:
            agent.llm_client = refreshed_client
    analytics = agent.analytics

    with st.sidebar:
        st.header("Dataset Overview")
        overview = analytics.dataset_overview or {}
        if overview:
            st.write(
                f"Rows: {overview.get('rows', '?')} | Customers: {overview.get('unique_customers', '?')}"
            )
            st.write(
                f"Products: {overview.get('unique_products', '?')}"
            )
            if overview.get("date_range"):
                st.write(
                    "Date range: "
                    f"{overview['date_range'].get('min', '?')} to {overview['date_range'].get('max', '?')}"
                )
        else:
            st.info("Run customer_query_agent.py to generate preprocessing artifacts first.")

        st.markdown("---")
        st.subheader("Sample Prompts")
        samples = [case["question"] for case in BENCHMARK_CASES]
        preset = st.selectbox("Choose a sample question", options=["", *samples])

    prompt = st.text_area(
        "Ask a question",
        value=preset or "Which products does Customer CUST_015 frequently purchase in the Beverages category?",
        height=120,
    )

    col_a, col_b = st.columns(2)
    run_answer = col_a.button("Run agent")
    show_retrieval = col_b.button("Show retrieval only")

    if run_answer and not prompt.strip():
        st.warning("Please enter a question before running the agent.")
        return

    if show_retrieval and not prompt.strip():
        st.warning("Please enter a question before inspecting retrieval.")
        return

    if show_retrieval:
        st.subheader("Top retrieved records")
        scored = analytics.retrieve_records(prompt, top_k=5)
        rows = [
            {
                "record_id": record.record_id,
                "similarity": round(score, 3),
                "text": record.text,
            }
            for score, record in scored
        ]
        render_retrieval(rows)
        return

    if run_answer:
        with st.spinner("Generating answer..."):
            result = agent.answer(prompt)
        st.subheader("Answer")
        st.write(result.get("answer", "No answer produced."))
        if result.get("reasoning"):
            st.caption(result["reasoning"])
        if result.get("llm_model"):
            st.caption(f"Generated with: {result['llm_model']}")

        render_retrieval(result.get("retrieved_records", []))
        render_supporting_facts(result.get("supporting_facts", []))

        if analytics.monthly_revenue:
            st.markdown("---")
            revenue_points = sorted(analytics.monthly_revenue.items())
            revenue_df = pd.DataFrame(revenue_points, columns=["month", "revenue"]).set_index("month")
            st.line_chart(revenue_df)

    st.markdown("---")
    st.caption("Streamlit UI backed by retrieval-augmented customer analytics.")


if __name__ == "__main__":
    main()
