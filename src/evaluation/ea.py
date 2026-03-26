"""Execution Accuracy metric."""

from __future__ import annotations

from typing import Any
from pathlib import Path

from src.evaluation.executor import ExecutionResult, execute_sql


def _results_match(pred_result: ExecutionResult, gold_result: ExecutionResult) -> bool:
    """Return True when both results succeeded with identical normalized rows."""
    return pred_result.success and gold_result.success and pred_result.rows == gold_result.rows


def execution_match(pred_sql: str, gold_result: ExecutionResult, db_path: Path, timeout: int = 30) -> bool:
    """Compare one prediction against a precomputed gold execution result."""
    pred_result = execute_sql(pred_sql, db_path, timeout=timeout)
    return _results_match(pred_result, gold_result)


def evaluate_candidate_predictions(
    predictions: list[str],
    gold_sql: str,
    db_path: Path,
    timeout: int = 30,
) -> dict[str, Any]:
    """Evaluate candidate SQL predictions against one gold query.

    Returns sample-level details that can be persisted and reused by later
    analysis steps without re-executing SQL in notebooks.
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
    """Evaluate multiple predictions while executing the gold SQL only once."""
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
    """Compute execution accuracy over aligned predictions and gold SQL."""
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
