"""Метрика Execution Accuracy."""

from __future__ import annotations

from typing import Any
from pathlib import Path

from nl2sql.src.evaluation.executor import ExecutionResult, execute_sql


def _results_match(pred_result: ExecutionResult, gold_result: ExecutionResult) -> bool:
    """Вернуть `True`, если оба результата успешны и совпадают после нормализации."""
    return pred_result.success and gold_result.success and pred_result.rows == gold_result.rows


def execution_match(pred_sql: str, gold_result: ExecutionResult, db_path: Path, timeout: int = 30) -> bool:
    """Сравнить одно предсказание с заранее вычисленным gold execution result."""
    pred_result = execute_sql(pred_sql, db_path, timeout=timeout)
    return _results_match(pred_result, gold_result)


def evaluate_candidate_predictions(
    predictions: list[str],
    gold_sql: str,
    db_path: Path,
    timeout: int = 30,
) -> dict[str, Any]:
    """Оценить SQL-кандидаты относительно одного gold-запроса.

    Возвращает sample-level детали, которые можно сохранить и потом переиспользовать.
    """
    gold_result = execute_sql(gold_sql, db_path, timeout=timeout)
    candidate_hits: list[bool] = []
    first_pred_error: str | None = None
    first_pred_success = False

    for idx, pred_sql in enumerate(predictions):
        pred_result = execute_sql(pred_sql, db_path, timeout=timeout)
        candidate_hits.append(_results_match(pred_result, gold_result))
        if idx == 0:
            first_pred_success = pred_result.success
            first_pred_error = pred_result.error

    return {
        "gold_success": gold_result.success,
        "gold_error": gold_result.error,
        "candidate_hits": candidate_hits,
        "first_pred_success": first_pred_success,
        "first_pred_error": first_pred_error,
    }


def candidate_execution_matches(
    predictions: list[str],
    gold_sql: str,
    db_path: Path,
    timeout: int = 30,
) -> list[bool]:
    """Оценить несколько предсказаний, исполнив gold SQL только один раз."""
    return evaluate_candidate_predictions(
        predictions,
        gold_sql,
        db_path,
        timeout=timeout,
    )["candidate_hits"]


def execution_accuracy(
    predictions: list[str],
    gold: list[str],
    db_paths: list[Path],
    timeout: int = 30,
) -> float:
    """Посчитать execution accuracy по выровненным спискам predictions и gold SQL."""
    if not (len(predictions) == len(gold) == len(db_paths)):
        raise ValueError("predictions, gold, and db_paths must have the same length")
    if not predictions:
        return 0.0

    matches = 0
    for pred_sql, gold_sql, db_path in zip(predictions, gold, db_paths, strict=True):
        gold_result = execute_sql(gold_sql, db_path, timeout=timeout)
        if execution_match(pred_sql, gold_result, db_path, timeout=timeout):
            matches += 1
    return matches / len(predictions)
