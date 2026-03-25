"""Execution Accuracy metric."""

from __future__ import annotations

from pathlib import Path

from src.evaluation.executor import ExecutionResult, execute_sql


def _results_match(pred_result: ExecutionResult, gold_result: ExecutionResult) -> bool:
    """Return True when both results succeeded with identical normalized rows."""
    return pred_result.success and gold_result.success and pred_result.rows == gold_result.rows


def execution_match(pred_sql: str, gold_result: ExecutionResult, db_path: Path, timeout: int = 30) -> bool:
    """Compare one prediction against a precomputed gold execution result."""
    pred_result = execute_sql(pred_sql, db_path, timeout=timeout)
    return _results_match(pred_result, gold_result)


def candidate_execution_matches(
    predictions: list[str],
    gold_sql: str,
    db_path: Path,
    timeout: int = 30,
) -> list[bool]:
    """Evaluate multiple predictions while executing the gold SQL only once."""
    gold_result = execute_sql(gold_sql, db_path, timeout=timeout)
    return [execution_match(pred_sql, gold_result, db_path, timeout=timeout) for pred_sql in predictions]


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
