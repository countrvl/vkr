import sqlite3
from pathlib import Path

from nl2sql.src.strategy_bench.dataset import TestCase
from nl2sql.src.strategy_bench.executor import SQLiteExecutor
from nl2sql.src.strategy_bench.validation import ValidationModule


def _build_db(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE users (id INTEGER, name TEXT)")
        connection.executemany("INSERT INTO users (id, name) VALUES (?, ?)", [(1, "Alice"), (2, "Bob")])
        connection.commit()


def test_validation_rejects_empty_and_non_read_only_sql(tmp_path: Path) -> None:
    db_path = tmp_path / "demo.sqlite"
    _build_db(db_path)
    executor = SQLiteExecutor(db_path)
    validator = ValidationModule()
    case = TestCase(id="case-1", natural_language_query="Count users", expected_sql="SELECT COUNT(*) FROM users")

    empty = validator.validate(case, "", executor)
    forbidden = validator.validate(case, "DELETE FROM users", executor)

    assert empty.success is False
    assert empty.error_type == "empty_sql"
    assert forbidden.success is False
    assert forbidden.error_type == "non_read_only"


def test_validation_reports_syntax_and_execution_failures(tmp_path: Path) -> None:
    db_path = tmp_path / "demo.sqlite"
    _build_db(db_path)
    executor = SQLiteExecutor(db_path)
    validator = ValidationModule()
    case = TestCase(id="case-1", natural_language_query="Count users", expected_sql="SELECT COUNT(*) FROM users")

    syntax = validator.validate(case, "SELECT FROM users", executor)
    runtime = validator.validate(case, "SELECT missing FROM users", executor)

    assert syntax.success is False
    assert syntax.error_type == "syntax_error"
    assert runtime.success is False
    assert runtime.error_type == "OperationalError"


def test_validation_compares_expected_result(tmp_path: Path) -> None:
    db_path = tmp_path / "demo.sqlite"
    _build_db(db_path)
    executor = SQLiteExecutor(db_path)
    validator = ValidationModule()
    case = TestCase(
        id="case-1",
        natural_language_query="List users",
        expected_result=[[1, "Alice"], [2, "Bob"]],
    )

    result = validator.validate(case, "SELECT id, name FROM users", executor)

    assert result.success is True
    assert result.accuracy is True
    assert result.comparison_mode == "expected_result"


def test_validation_uses_expected_sql_result_and_excludes_invalid_reference(tmp_path: Path) -> None:
    db_path = tmp_path / "demo.sqlite"
    _build_db(db_path)
    executor = SQLiteExecutor(db_path)
    validator = ValidationModule()
    valid_case = TestCase(
        id="case-1",
        natural_language_query="Count users",
        expected_sql="SELECT COUNT(*) FROM users",
    )
    invalid_case = TestCase(
        id="case-2",
        natural_language_query="Count users",
        expected_sql="SELECT nope FROM users",
    )

    valid_result = validator.validate(valid_case, "SELECT COUNT(*) FROM users", executor)
    invalid_result = validator.validate(invalid_case, "SELECT COUNT(*) FROM users", executor)

    assert valid_result.accuracy is True
    assert valid_result.comparison_mode == "expected_sql_result"
    assert invalid_result.success is True
    assert invalid_result.invalid_reference is True
    assert invalid_result.accuracy is None
