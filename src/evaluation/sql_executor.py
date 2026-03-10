"""SQL execution utilities built on sqlite3."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def execute_sql(db_path: str | Path, sql: str) -> list[tuple[Any, ...]]:
    """Execute SQL on SQLite and return normalized row tuples.

    Args:
        db_path: Path to SQLite database file.
        sql: SQL query to execute.

    Returns:
        Query results sorted for stable comparison.

    Raises:
        FileNotFoundError: If DB does not exist.
        sqlite3.Error: For invalid SQL or execution failures.
    """
    db = Path(db_path)
    if not db.exists():
        raise FileNotFoundError(f"SQLite database not found: {db}")

    with sqlite3.connect(str(db)) as connection:
        cursor = connection.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()

    normalized = [tuple(_normalize_value(value) for value in row) for row in rows]
    return sorted(normalized)


def _normalize_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 8)
    return value
