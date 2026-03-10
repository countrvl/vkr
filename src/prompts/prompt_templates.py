"""Prompt templates for NL2SQL generation."""

from __future__ import annotations


def build_nl2sql_prompt(question: str, schema: str) -> str:
    """Build a deterministic NL2SQL prompt.

    Args:
        question: Natural language question.
        schema: Database schema description.

    Returns:
        Prompt string instructing the model to generate SQL only.
    """
    return (
        "You are an expert SQL assistant.\\n"
        "Generate a syntactically correct SQLite SQL query for the question.\\n"
        "Return only SQL, no explanation.\\n\\n"
        f"Schema:\\n{schema}\\n\\n"
        f"Question: {question}\\n"
        "SQL:"
    )
