"""Execution layer for production-like strategy evaluation."""

from __future__ import annotations

import re
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_FORBIDDEN_KEYWORDS = (
    "insert",
    "update",
    "delete",
    "alter",
    "drop",
    "create",
    "truncate",
    "copy",
    "call",
    "do",
)
_ALLOWED_PREFIXES = ("select", "with")
_SQL_COMMENT_PATTERN = re.compile(r"(--[^\n]*$)|(/\*.*?\*/)", flags=re.MULTILINE | re.DOTALL)


@dataclass(slots=True)
class ValidationIssue:
    """Validation problem identified before or during execution."""

    code: str
    message: str


@dataclass(slots=True)
class ExecutionOutcome:
    """Normalized outcome of SQL execution."""

    success: bool
    rows: list[tuple[str, ...]] | None
    error_type: str | None = None
    error_message: str | None = None
    latency_ms: float = 0.0


@dataclass(slots=True)
class SchemaContext:
    """Prompt-friendly schema snapshot."""

    dialect: str
    text: str


class SqlExecutor(ABC):
    """Abstract execution backend."""

    @abstractmethod
    def execute(self, sql: str) -> ExecutionOutcome:
        """Execute SQL and return normalized rows or an error."""

    @abstractmethod
    def explain_syntax(self, sql: str) -> ValidationIssue | None:
        """Validate syntax using a safe precheck."""

    @abstractmethod
    def get_schema_context(self) -> SchemaContext:
        """Return prompt-ready schema context."""


def normalize_value(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def normalize_rows(rows: list[tuple[Any, ...]]) -> list[tuple[str, ...]]:
    normalized = [tuple(normalize_value(cell) for cell in row) for row in rows]
    return sorted(normalized)


def normalize_expected_result(value: list[list[Any]] | list[dict[str, Any]]) -> list[tuple[str, ...]]:
    normalized_rows: list[tuple[str, ...]] = []
    for row in value:
        if isinstance(row, dict):
            normalized_rows.append(tuple(normalize_value(row[key]) for key in sorted(row)))
        else:
            normalized_rows.append(tuple(normalize_value(cell) for cell in row))
    return sorted(normalized_rows)


def guard_read_only_sql(sql: str) -> ValidationIssue | None:
    stripped = sql.strip()
    if not stripped:
        return ValidationIssue(code="empty_sql", message="SQL query is empty")
    without_comments = _SQL_COMMENT_PATTERN.sub("", stripped).strip()
    lowered = without_comments.lower()
    if not lowered.startswith(_ALLOWED_PREFIXES):
        return ValidationIssue(code="non_read_only", message="Only SELECT/with queries are allowed")
    trimmed = without_comments[:-1] if without_comments.endswith(";") else without_comments
    if ";" in trimmed:
        return ValidationIssue(code="multiple_statements", message="Multiple SQL statements are not allowed")
    if re.search(r"\b(" + "|".join(_FORBIDDEN_KEYWORDS) + r")\b", lowered):
        return ValidationIssue(code="forbidden_keyword", message="Forbidden SQL keyword in read-only mode")
    return None


class PostgresExecutor(SqlExecutor):
    """Read-only PostgreSQL executor with lazy schema caching."""

    def __init__(self, dsn: str, *, schema: str = "public") -> None:
        self._dsn = dsn
        self._schema = schema
        self._schema_context: SchemaContext | None = None
        try:
            import psycopg  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError(
                "PostgresExecutor requires psycopg>=3. Install it before running the strategy bench."
            ) from exc
        self._psycopg = psycopg

    def _connect(self):
        connection = self._psycopg.connect(self._dsn, autocommit=True)
        with connection.cursor() as cursor:
            cursor.execute("SET default_transaction_read_only = on")
        return connection

    def execute(self, sql: str) -> ExecutionOutcome:
        issue = guard_read_only_sql(sql)
        if issue is not None:
            return ExecutionOutcome(
                success=False,
                rows=None,
                error_type=issue.code,
                error_message=issue.message,
            )
        started_at = time.perf_counter()
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
        except Exception as exc:  # pragma: no cover - depends on runtime DB
            return ExecutionOutcome(
                success=False,
                rows=None,
                error_type=type(exc).__name__,
                error_message=str(exc),
                latency_ms=(time.perf_counter() - started_at) * 1000.0,
            )
        return ExecutionOutcome(
            success=True,
            rows=normalize_rows([tuple(row) for row in rows]),
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
        )

    def explain_syntax(self, sql: str) -> ValidationIssue | None:
        issue = guard_read_only_sql(sql)
        if issue is not None:
            return issue
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(f"EXPLAIN {sql}")
        except Exception as exc:  # pragma: no cover - depends on runtime DB
            code = "syntax_error" if "syntax" in str(exc).lower() else type(exc).__name__
            return ValidationIssue(code=code, message=str(exc))
        return None

    def get_schema_context(self) -> SchemaContext:
        if self._schema_context is not None:
            return self._schema_context
        query = """
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = %s
            ORDER BY table_name, ordinal_position
        """
        table_map: dict[str, list[str]] = {}
        with self._connect() as connection, connection.cursor() as cursor:  # pragma: no cover - runtime DB
            cursor.execute(query, (self._schema,))
            for table_name, column_name, data_type in cursor.fetchall():
                table_map.setdefault(str(table_name), []).append(f"{column_name} {data_type}")
        lines = ["Dialect: PostgreSQL", f"Schema: {self._schema}"]
        for table_name, columns in table_map.items():
            lines.append(f"TABLE {table_name} ({', '.join(columns)})")
        self._schema_context = SchemaContext(dialect="postgresql", text="\n".join(lines))
        return self._schema_context


class SQLiteExecutor(SqlExecutor):
    """SQLite executor used in tests and local smoke harnesses."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._schema_context: SchemaContext | None = None

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def execute(self, sql: str) -> ExecutionOutcome:
        issue = guard_read_only_sql(sql)
        if issue is not None:
            return ExecutionOutcome(
                success=False,
                rows=None,
                error_type=issue.code,
                error_message=issue.message,
            )
        started_at = time.perf_counter()
        try:
            with self._connect() as connection:
                rows = connection.execute(sql).fetchall()
        except sqlite3.DatabaseError as exc:
            return ExecutionOutcome(
                success=False,
                rows=None,
                error_type=type(exc).__name__,
                error_message=str(exc),
                latency_ms=(time.perf_counter() - started_at) * 1000.0,
            )
        return ExecutionOutcome(
            success=True,
            rows=normalize_rows([tuple(row) for row in rows]),
            latency_ms=(time.perf_counter() - started_at) * 1000.0,
        )

    def explain_syntax(self, sql: str) -> ValidationIssue | None:
        issue = guard_read_only_sql(sql)
        if issue is not None:
            return issue
        try:
            with self._connect() as connection:
                connection.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()
        except sqlite3.DatabaseError as exc:
            code = "syntax_error" if "syntax" in str(exc).lower() else type(exc).__name__
            return ValidationIssue(code=code, message=str(exc))
        return None

    def get_schema_context(self) -> SchemaContext:
        if self._schema_context is not None:
            return self._schema_context
        lines = ["Dialect: SQLite"]
        with self._connect() as connection:
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
            for row in tables:
                table_name = str(row["name"])
                columns = connection.execute(f"PRAGMA table_info('{table_name}')").fetchall()
                column_defs = [f"{col['name']} {col['type']}" for col in columns]
                lines.append(f"TABLE {table_name} ({', '.join(column_defs)})")
        self._schema_context = SchemaContext(dialect="sqlite", text="\n".join(lines))
        return self._schema_context
