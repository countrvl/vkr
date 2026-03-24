"""SQLite execution with normalization and timeout handling."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExecutionResult:
    """Outcome of executing a SQL query."""

    success: bool
    rows: list[tuple[str, ...]] | None
    error: str | None = None


def _normalize_value(value: Any) -> str:
    """Normalize a single cell value to a canonical string.

    Floats are rendered with 10 significant digits (``.10g``) to tolerate
    floating-point representation differences while preserving enough precision
    to detect genuine numeric mismatches.  NULL is mapped to the literal string
    ``"NULL"``; all other types use ``str()``.
    """
    if value is None:
        return "NULL"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _normalize_rows(rows: list[tuple[Any, ...]]) -> list[tuple[str, ...]]:
    """Normalize and sort result rows for order-independent comparison.

    Row order is intentionally discarded: Spider and BIRD evaluate Execution
    Accuracy by result-*set* equality, not sequence equality.  This is the
    standard treatment in the benchmark literature — queries with ``ORDER BY``
    are still considered correct as long as their result set matches the gold
    result set.
    """
    normalized = [tuple(_normalize_value(value) for value in row) for row in rows]
    return sorted(normalized)


def execute_sql(sql: str, db_path: Path, timeout: int = 30) -> ExecutionResult:
    """Execute SQL against SQLite with a progress-handler timeout.

    Args:
        sql: SQL query to execute.
        db_path: SQLite database path.
        timeout: Timeout in seconds.

    Returns:
        Normalized execution result.
    """
    if not sql or not sql.strip():
        return ExecutionResult(success=False, rows=None, error="empty query")

    started_at = time.monotonic()
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row

        def progress_handler() -> int:
            return 1 if time.monotonic() - started_at > timeout else 0

        connection.set_progress_handler(progress_handler, 1_000)
        try:
            cursor = connection.execute(sql)
            rows = cursor.fetchall()
            normalized_rows = _normalize_rows([tuple(row) for row in rows])
            return ExecutionResult(success=True, rows=normalized_rows)
        except sqlite3.OperationalError as exc:
            message = str(exc)
            if "interrupted" in message.lower():
                return ExecutionResult(success=False, rows=None, error="timeout")
            return ExecutionResult(success=False, rows=None, error=message)
        except sqlite3.DatabaseError as exc:
            return ExecutionResult(success=False, rows=None, error=str(exc))
        finally:
            connection.set_progress_handler(None, 0)
