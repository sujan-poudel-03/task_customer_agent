"""Structured pandas pipeline that generates and executes LLM-authored queries.

Workflow
--------
1. Load the orders CSV into a pandas DataFrame and capture its schema.
2. Ask Groq to produce a single pandas expression tailored to the user question.
   The expression must assign its final output to a variable named ``result``.
3. Execute that query, print intermediate artefacts, and send the resulting
   preview back to Groq for the final natural-language answer.
"""
from __future__ import annotations

import argparse
import builtins
import datetime as dt
import json
import os
import traceback
import textwrap
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "Dataset_product_orders.csv"
GROQ_API_KEY_ENV = os.getenv("GROQ_API_KEY")

ALLOWED_BUILTINS: Dict[str, Any] = {
    name: getattr(builtins, name)
    for name in [
        "len",
        "range",
        "min",
        "max",
        "sum",
        "sorted",
        "list",
        "set",
        "dict",
        "tuple",
        "enumerate",
        "any",
        "all",
        "round",
        "abs",
        "zip",
        "map",
        "filter",
        "float",
        "int",
        "str",
        "bool",
    ]
}
ALLOWED_BUILTINS["__import__"] = builtins.__import__

SAFE_GLOBALS: Dict[str, Any] = {
    "pd": pd,
    "np": np,
    "Counter": Counter,
    "combinations": combinations,
    "__builtins__": ALLOWED_BUILTINS,
}


class GroqChatClient:
    """Thin wrapper around the Groq chat completions endpoint."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_completion_tokens: Optional[int] = None,
        top_p: float = 1.0,
        reasoning_effort: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set; update your .env file.")
        self.model = model or os.getenv("GROQ_CHAT_MODEL") or "openai/gpt-oss-20b"
        self.base_url = base_url or os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
        self.timeout = timeout
        self._session = requests.Session()
        self.default_max_completion_tokens = max_completion_tokens
        self.default_top_p = top_p
        self.default_reasoning_effort = reasoning_effort

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_completion_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }

        token_limit = max_completion_tokens or self.default_max_completion_tokens
        if token_limit is not None:
            payload["max_completion_tokens"] = int(token_limit)

        top_p_value = top_p if top_p is not None else self.default_top_p
        if top_p_value is not None:
            payload["top_p"] = float(top_p_value)

        reasoning_value = reasoning_effort or self.default_reasoning_effort
        if reasoning_value:
            payload["reasoning_effort"] = reasoning_value

        response = self._session.post(
            url,
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Groq chat request failed ({response.status_code}): {response.text}"
            )
        payload = response.json()
        choices = payload.get("choices")
        if not choices:
            raise RuntimeError("Unexpected chat response shape.")
        return choices[0]["message"]["content"]


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        if "```" in cleaned:
            cleaned = cleaned.split("```", 1)[0].strip()
    return cleaned


def _parse_query_response(raw: str) -> Dict[str, Any]:
    cleaned = _strip_code_fence(raw)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("Query generator returned invalid JSON.") from exc
    if "pandas_query" not in parsed or not isinstance(parsed["pandas_query"], str):
        raise ValueError("JSON response must include a 'pandas_query' string.")
    return parsed


def _serialise_value(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_serialise_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialise_value(v) for k, v in value.items()}
    return value


def _build_safe_environment(df: pd.DataFrame) -> tuple[Dict[str, Any], Dict[str, Any]]:
    safe_globals = dict(SAFE_GLOBALS)
    safe_locals = {"df": df.copy()}
    return safe_globals, safe_locals


def _prepare_result_preview(result: Any, limit: int = 20) -> Dict[str, Any]:
    if isinstance(result, pd.DataFrame):
        preview_df = result.head(limit)
        preview = [
            {key: _serialise_value(value) for key, value in row.items()}
            for row in preview_df.to_dict(orient="records")
        ]
        summary = {
            "type": "dataframe",
            "rows": int(result.shape[0]),
            "columns": list(result.columns.astype(str)),
        }
        return {"summary": summary, "preview": preview}

    if isinstance(result, pd.Series):
        preview_series = result.head(limit)
        preview = [
            {
                "index": _serialise_value(idx),
                "value": _serialise_value(val),
            }
            for idx, val in preview_series.items()
        ]
        summary = {
            "type": "series",
            "rows": int(result.shape[0]),
            "name": getattr(result, "name", None),
        }
        return {"summary": summary, "preview": preview}

    if isinstance(result, (list, tuple)):
        preview = [_serialise_value(entry) for entry in list(result)[:limit]]
        summary = {"type": "list", "length": len(result)}
        return {"summary": summary, "preview": preview}

    if isinstance(result, dict):
        items: Sequence[tuple[Any, Any]] = list(result.items())[:limit]
        preview = [
            {
                "key": _serialise_value(key),
                "value": _serialise_value(val),
            }
            for key, val in items
        ]
        summary = {"type": "dict", "length": len(result)}
        return {"summary": summary, "preview": preview}

    summary = {"type": type(result).__name__}
    return {"summary": summary, "preview": _serialise_value(result)}


def _print_result_preview(result: Any, limit: int = 20) -> None:
    print(f"[pipeline] Query result type: {type(result).__name__}")
    if isinstance(result, pd.DataFrame):
        print(result.head(limit).to_string(index=False))
    elif isinstance(result, pd.Series):
        print(result.head(limit).to_string())
    else:
        print(result)


@dataclass
class StructuredRAGPipeline:
    df: pd.DataFrame
    chat_client: GroqChatClient

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path = DEFAULT_DATASET,
        *,
        chat_client: GroqChatClient,
    ) -> "StructuredRAGPipeline":
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset not found at {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=["Date"], low_memory=False)
        df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
        print(f"[pipeline] Loaded dataset with shape {df.shape}")
        print("[pipeline] Columns:", ", ".join(df.columns))
        return cls(df=df, chat_client=chat_client)

    @property
    def columns(self) -> List[str]:
        return list(self.df.columns)

    def _schema_preview(self, max_values: int = 3) -> Dict[str, List[str]]:
        preview: Dict[str, List[str]] = {}
        for column in self.columns:
            series = self.df[column]
            values = (
                series.dropna()
                .astype(str)
                .map(lambda v: v.strip())
                .drop_duplicates()
                .head(max_values)
                .tolist()
            )
            preview[column] = values
        return preview

    def _rows_preview(self, limit: int = 5) -> List[Dict[str, Any]]:
        sample = self.df.head(limit).to_dict(orient="records")
        return [{key: _serialise_value(value) for key, value in row.items()} for row in sample]

    def generate_pandas_query(
        self,
        question: str,
        *,
        feedback: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        schema_lines = "\n".join(f"- {name}" for name in self.columns)
        preview_json = json.dumps(self._schema_preview(), indent=2)
        rows_json = json.dumps(self._rows_preview(), indent=2)
        feedback_section = ""
        if feedback:
            guidance = feedback.get(
                "hint",
                "Guidance: avoid using DataFrame.query with external variables; prefer boolean masks or define helper variables inline.",
            )
            feedback_section = textwrap.dedent(
                f"""
                Previous attempt:
                {feedback.get("pandas_query", "").strip()}

                Error raised:
                {feedback.get("error", "Unknown error")}

                {guidance}

                Revise the pandas query to address the error. Produce a fresh solution.
                """
            ).strip()

        system_prompt = (
            "You are a senior pandas architect. Craft a single pandas expression that solves the user's question.\n"
            "Constraints:\n"
            "- The DataFrame is already loaded as variable `df` with snake_case columns.\n"
            "- Use method chaining where possible.\n"
            "- Assign the final result to a variable named `result` (e.g. `result = ...`).\n"
            "- Avoid using `.query()` with external variables (`@var`). Prefer boolean indexing or define helper variables in the same expression.\n"
            "- You may rely on the helpers already imported for you: `pd`, `np`, `Counter`, and `combinations`.\n"
            "- Do not emit explanation text; respond with JSON only.\n"
            "Return JSON with keys:\n"
            "  * pandas_query: the code string.\n"
            "  * reasoning: optional short explanation of the approach.\n"
            "  * result_type: optional description of the expected shape (e.g. dataframe, series).\n"
        )

        user_prompt = textwrap.dedent(
            f"""
            Dataset columns (snake_case):
            {schema_lines}

            Sample categorical values:
            {preview_json}

            Sample rows:
            {rows_json}

            {feedback_section}

            Question:
            {question}

            Respond with JSON only.
            """
        ).strip()

        raw = self.chat_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
        )
        return _parse_query_response(raw)

    def _execute_query(self, pandas_query: str) -> Any:
        if not pandas_query or "result" not in pandas_query:
            raise ValueError("Generated pandas_query must assign to a variable named 'result'.")
        safe_globals, safe_locals = _build_safe_environment(self.df)
        exec(pandas_query, safe_globals, safe_locals)
        if "result" not in safe_locals:
            raise ValueError("Pandas query did not produce a variable named 'result'.")
        return safe_locals["result"]

    def answer_question(
        self,
        question: str,
        *,
        preview_limit: int = 20,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        feedback: Optional[Dict[str, str]] = None
        query_spec: Optional[Dict[str, Any]] = None
        pandas_query = ""
        result: Any = None
        last_exc: Optional[Exception] = None
        last_error_text = ""

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            print(f"[pipeline] Generating pandas query (attempt {attempt}/{max_attempts})...")
            query_spec = self.generate_pandas_query(question, feedback=feedback)
            pandas_query = query_spec.get("pandas_query", "")
            print("[pipeline] Query generator response:")
            print(json.dumps(query_spec, indent=2))
            print("[pipeline] Pandas query:\n" + pandas_query)

            try:
                print("[pipeline] Executing pandas query...")
                result = self._execute_query(pandas_query)
                _print_result_preview(result, limit=preview_limit)
                break
            except Exception as exc:
                last_exc = exc
                last_error_text = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                print(f"[pipeline] Query execution failed: {last_error_text}")
                hint = ""
                if "UndefinedVariableError" in last_error_text:
                    hint = (
                        "Hint: avoid relying on DataFrame.query with external variables. "
                        "Use boolean masks or define variables inside the expression."
                    )
                feedback = {
                    "pandas_query": pandas_query,
                    "error": last_error_text,
                    "hint": hint,
                }
        else:
            raise RuntimeError(
                f"Generated pandas query failed after {max_attempts} attempts: {last_error_text}"
            ) from last_exc

        result_payload = _prepare_result_preview(result, limit=preview_limit)

        analysis_payload = {
            "pandas_query": pandas_query,
            "intent_reasoning": query_spec.get("reasoning"),
            "result_overview": result_payload["summary"],
            "result_preview": result_payload["preview"],
        }
        analysis_json = json.dumps(analysis_payload, indent=2, default=_serialise_value)

        system_prompt = (
            "You are an analytics assistant. Review the provided pandas query and its result preview, "
            "then answer the business question. Only rely on the supplied data. "
            "If the result is empty or not relevant, state that clearly."
        )
        user_prompt = textwrap.dedent(
            f"""
            Question: {question}

            Pandas query that was executed:
            {pandas_query}

            Query rationale: {query_spec.get("reasoning", "N/A")}

            Result summary and preview:
            {analysis_json}

            Provide a concise answer grounded in the result preview.
            """
        ).strip()

        answer = self.chat_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )

        return {
            "question": question,
            "pandas_query": pandas_query,
            "query_reasoning": query_spec.get("reasoning"),
            "result_overview": result_payload["summary"],
            "result_preview": result_payload["preview"],
            "answer": answer.strip(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Structured pandas analytics via Groq-generated queries.")
    parser.add_argument(
        "--question",
        type=str,
        help="Natural language question to answer using the orders dataset.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=20,
        help="Number of rows/items to include in the result preview (default: 20).",
    )
    args = parser.parse_args()

    question = args.question
    if not question:
        question = input("Enter your question: ").strip()
    if not question:
        raise SystemExit("Question is required.")

    chat_client = GroqChatClient(api_key=GROQ_API_KEY_ENV)
    pipeline = StructuredRAGPipeline.from_csv(chat_client=chat_client)

    result = pipeline.answer_question(
        question,
        preview_limit=args.preview,
    )
    print("-----")
    print("Pandas query executed:")
    print(result["pandas_query"])
    print("Answer:")
    print(result["answer"])


if __name__ == "__main__":
    main()
