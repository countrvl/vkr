"""Execution Accuracy metric."""

from __future__ import annotations

from pathlib import Path

from src.evaluation.executor import execute_sql


def execution_accuracy(predictions: list[str], gold: list[str], db_paths: list[Path]) -> float:
    """Compute execution accuracy over aligned predictions and gold SQL."""
    if not (len(predictions) == len(gold) == len(db_paths)):
        raise ValueError("predictions, gold, and db_paths must have the same length")
    if not predictions:
        return 0.0

    matches = 0
    for pred_sql, gold_sql, db_path in zip(predictions, gold, db_paths, strict=True):
        pred_result = execute_sql(pred_sql, db_path)
        gold_result = execute_sql(gold_sql, db_path)
        if pred_result.success and gold_result.success and pred_result.rows == gold_result.rows:
            matches += 1
    return matches / len(predictions)
