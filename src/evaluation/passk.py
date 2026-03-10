"""Pass@K metric for multiple SQL generations per query."""

from __future__ import annotations

from pathlib import Path

from src.evaluation.execution_accuracy import execution_match


def pass_at_k(db_path: str | Path, gold_sql: str, predictions: list[str], k: int) -> bool:
    """Return True if any of top-k predictions matches gold execution."""
    if k <= 0:
        raise ValueError("k must be positive")

    top_k = predictions[:k]
    return any(execution_match(db_path, gold_sql, pred) for pred in top_k)


def compute_pass_at_k(records: list[dict], k: int) -> float:
    """Compute dataset-level pass@k from per-record pass flags."""
    if not records:
        return 0.0

    key = f"pass_at_{k}"
    passed = sum(1 for row in records if bool(row.get(key, False)))
    return passed / len(records)
