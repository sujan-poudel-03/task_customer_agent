"""Minimal Streamlit interface for Groq-driven pandas analytics."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

try:  # accommodate running from repo root or task subdir
    from task.structured_rag import DEFAULT_DATASET, GroqChatClient, StructuredRAGPipeline
except ModuleNotFoundError:  # pragma: no cover - fallback for direct execution
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from structured_rag import DEFAULT_DATASET, GroqChatClient, StructuredRAGPipeline


def ensure_env() -> None:
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is missing. Update your .env file before running the app.")
        st.stop()


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DEFAULT_DATASET, parse_dates=["Date"], low_memory=False)
    df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
    return df


def main() -> None:
    st.set_page_config(page_title="Groq Pandas Analytics", layout="wide")
    st.title("Structured Customer Analytics - Groq-powered pandas")
    ensure_env()

    question = st.text_area(
        "Ask a question about the orders dataset",
        value="Find frequently co-purchased product pairs among high-value customers.",
        height=120,
    )
    # temperature = st.slider("LLM temperature", 0.0, 1.0, 0.2, 0.05)
    # preview_limit = st.slider("Preview rows/items", 5, 50, 20, 5)

    if not st.button("Run analysis"):
        return

    if not question.strip():
        st.warning("Please enter a question before running the analysis.")
        return

    try:
        chat_client = GroqChatClient()
    except RuntimeError as exc:
        st.error(f"Failed to initialise Groq client: {exc}")
        return

    try:
        df = load_dataset()
    except Exception as exc:
        st.error(f"Failed to load dataset: {exc}")
        return

    pipeline = StructuredRAGPipeline(df=df, chat_client=chat_client)

    with st.spinner("Generating pandas query and running analysis..."):
        try:
            result = pipeline.answer_question(
                question,
                # preview_limit=preview_limit,
                # temperature=temperature,
            )
        except Exception as exc:  # broad to surface in UI
            st.error(f"Pipeline execution failed: {exc}")
            return

    st.subheader("Answer")
    st.write(result.get("answer", "No answer generated."))

    st.subheader("Generated pandas query")
    st.code(result.get("pandas_query", ""), language="python")

    reasoning = result.get("query_reasoning")
    if reasoning:
        st.info(f"Query rationale: {reasoning}")

    summary = result.get("result_overview")
    preview = result.get("result_preview")

    if summary:
        st.subheader("Result summary")
        st.json(summary)

    if preview is not None:
        st.subheader("Result preview")
        if isinstance(preview, list):
            st.dataframe(pd.DataFrame(preview))
        else:
            st.write(preview)


if __name__ == "__main__":
    main()
