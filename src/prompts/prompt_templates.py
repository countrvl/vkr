"""Шаблоны промптов для генерации NL2SQL."""

from __future__ import annotations


def build_nl2sql_prompt(question: str, schema: str) -> str:
    """Собрать детерминированный промпт для NL2SQL.

    Args:
        question: Вопрос на естественном языке.
        schema: Описание схемы базы данных.

    Returns:
        Строка промпта с инструкцией вернуть только SQL.
    """
    return (
        "You are an expert SQL assistant.\n"
        "Generate a syntactically correct SQLite SQL query for the question.\n"
        "Return only SQL, no explanation.\n\n"
        f"Schema:\n{schema}\n\n"
        f"Question: {question}\n"
        "SQL:"
    )
