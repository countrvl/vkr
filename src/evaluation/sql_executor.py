"""Утилиты выполнения SQL на базе sqlite3."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def execute_sql(db_path: str | Path, sql: str) -> list[tuple[Any, ...]]:
    """Выполнить SQL в SQLite и вернуть нормализованные строки результата.

    Args:
        db_path: Путь к SQLite-файлу базы данных.
        sql: SQL-запрос для выполнения.

    Returns:
        Отсортированные строки результата для стабильного сравнения.

    Raises:
        FileNotFoundError: Если файл базы не существует.
        sqlite3.Error: Если SQL некорректен или выполнение завершилось ошибкой.
    """
    db = Path(db_path)
    if not db.exists():
        raise FileNotFoundError(f"SQLite база не найдена: {db}")

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
