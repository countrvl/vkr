"""Выполнение SQL в SQLite с нормализацией и таймаутом."""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CONNECTION_CACHE: dict[str, sqlite3.Connection] = {}
_RESULT_CACHE: dict[tuple[str, str, int], "ExecutionResult"] = {}
_MAX_RESULT_CACHE_SIZE = 10_000
_SQLITE_MMAP_SIZE_BYTES = 256 * 1024 * 1024
_SQLITE_CACHE_SIZE_PAGES_KB = -200_000


@dataclass(slots=True)
class ExecutionResult:
    """Результат выполнения SQL-запроса."""

    success: bool
    rows: list[tuple[str, ...]] | None
    error: str | None = None


def _normalize_value(value: Any) -> str:
    """Нормализовать одно значение ячейки в каноническую строку."""
    if value is None:
        return "NULL"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _normalize_rows(rows: list[tuple[Any, ...]]) -> list[tuple[str, ...]]:
    """Нормализовать и отсортировать строки результата для сравнения без учета порядка."""
    normalized = [tuple(_normalize_value(value) for value in row) for row in rows]
    return sorted(normalized)


def _cache_key(sql: str, db_path: Path, timeout: int) -> tuple[str, str, int]:
    return (str(db_path.resolve()), sql, timeout)


def _get_connection(db_path: Path) -> sqlite3.Connection:
    cache_key = str(db_path.resolve())
    connection = _CONNECTION_CACHE.get(cache_key)
    if connection is None:
        connection = sqlite3.connect(
            f"file:{db_path.resolve()}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        # Evaluation is read-only, so prefer memory-backed temp storage and a larger page cache.
        # This substantially reduces the number of pass@k candidates that fall into the timeout path.
        pragmas = (
            "PRAGMA query_only=ON",
            "PRAGMA temp_store=MEMORY",
            f"PRAGMA cache_size={_SQLITE_CACHE_SIZE_PAGES_KB}",
            f"PRAGMA mmap_size={_SQLITE_MMAP_SIZE_BYTES}",
            "PRAGMA journal_mode=OFF",
            "PRAGMA synchronous=OFF",
        )
        for pragma in pragmas:
            try:
                connection.execute(pragma)
            except sqlite3.DatabaseError:
                # Some bundled benchmark DBs reject a subset of tuning PRAGMAs
                # (for example in read-only or connector-specific layouts).
                # The evaluator should keep working with the remaining settings.
                continue
        _CONNECTION_CACHE[cache_key] = connection
    return connection


def clear_executor_caches() -> None:
    """Очистить process-local кэши результатов SQL и соединений."""
    for connection in _CONNECTION_CACHE.values():
        connection.close()
    _CONNECTION_CACHE.clear()
    _RESULT_CACHE.clear()


def execute_sql(sql: str, db_path: Path, timeout: int = 30) -> ExecutionResult:
    """Выполнить SQL в SQLite с таймаутом через progress handler."""
    if not sql or not sql.strip():
        return ExecutionResult(success=False, rows=None, error="empty query")

    key = _cache_key(sql, db_path, timeout)
    cached = _RESULT_CACHE.get(key)
    if cached is not None:
        return cached

    started_at = time.monotonic()
    connection = _get_connection(db_path)
    watchdog = threading.Timer(timeout, connection.interrupt)
    watchdog.daemon = True

    def progress_handler() -> int:
        return 1 if time.monotonic() - started_at > timeout else 0

    connection.set_progress_handler(progress_handler, 1_000)
    watchdog.start()
    try:
        cursor = connection.execute(sql)
        rows = cursor.fetchall()
        result = ExecutionResult(success=True, rows=_normalize_rows([tuple(row) for row in rows]))
    except sqlite3.OperationalError as exc:
        message = str(exc)
        if "interrupted" in message.lower():
            result = ExecutionResult(success=False, rows=None, error="timeout")
        else:
            result = ExecutionResult(success=False, rows=None, error=message)
    except sqlite3.DatabaseError as exc:
        result = ExecutionResult(success=False, rows=None, error=str(exc))
    finally:
        watchdog.cancel()
        connection.set_progress_handler(None, 0)

    if len(_RESULT_CACHE) >= _MAX_RESULT_CACHE_SIZE:
        evict_keys = list(_RESULT_CACHE)[:_MAX_RESULT_CACHE_SIZE // 2]
        for evict_key in evict_keys:
            del _RESULT_CACHE[evict_key]
    _RESULT_CACHE[key] = result
    return result
