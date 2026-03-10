"""Execution Accuracy evaluation for NL2SQL outputs."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from src.evaluation.sql_executor import execute_sql


def execution_match(db_path: str | Path, gold_sql: str, predicted_sql: str) -> bool:
    """Compare execution results of gold and predicted SQL on SQLite."""
    try:
        gold_rows = execute_sql(db_path, gold_sql)
        pred_rows = execute_sql(db_path, predicted_sql)
    except (sqlite3.Error, FileNotFoundError):
        return False
    return gold_rows == pred_rows


def compute_execution_accuracy(records: list[dict[str, Any]]) -> float:
    """Compute execution accuracy over evaluated records.

    Each record is expected to contain `is_correct: bool`.
    """
    if not records:
        return 0.0

    correct = sum(1 for row in records if bool(row.get("is_correct", False)))
    return correct / len(records)
