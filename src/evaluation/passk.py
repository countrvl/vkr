"""Метрика Pass@K для нескольких генераций SQL на один запрос."""

from __future__ import annotations

from pathlib import Path

from src.evaluation.execution_accuracy import execution_match


def pass_at_k(db_path: str | Path, gold_sql: str, predictions: list[str], k: int) -> bool:
    """Вернуть True, если хотя бы один из top-k кандидатов совпал с gold по выполнению."""
    if k <= 0:
        raise ValueError("k должен быть положительным")

    top_k = predictions[:k]
    return any(execution_match(db_path, gold_sql, pred) for pred in top_k)


def compute_pass_at_k(records: list[dict], k: int) -> float:
    """Посчитать Pass@K по датасету из флагов по каждой записи."""
    if not records:
        return 0.0

    key = f"pass_at_{k}"
    passed = sum(1 for row in records if bool(row.get(key, False)))
    return passed / len(records)
