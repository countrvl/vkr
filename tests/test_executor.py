import sqlite3
from pathlib import Path

from src.evaluation.executor import execute_sql


def _build_db(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE users (id INTEGER, name TEXT)")
        connection.executemany(
            "INSERT INTO users (id, name) VALUES (?, ?)",
            [(2, "Bob"), (1, "Alice")],
        )
        connection.commit()


def test_execute_sql_normalizes_row_order(tmp_path: Path) -> None:
    db_path = tmp_path / "demo.sqlite"
    _build_db(db_path)
    result = execute_sql("SELECT id, name FROM users ORDER BY id DESC", db_path)
    assert result.success is True
    assert result.rows == [("1", "Alice"), ("2", "Bob")]


def test_execute_sql_reports_errors(tmp_path: Path) -> None:
    db_path = tmp_path / "demo.sqlite"
    _build_db(db_path)
    result = execute_sql("SELECT missing FROM users", db_path)
    assert result.success is False
    assert result.error is not None


def test_execute_sql_timeout_path(tmp_path: Path) -> None:
    db_path = tmp_path / "demo.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE nums (value INTEGER)")
        connection.executemany("INSERT INTO nums (value) VALUES (?)", [(i,) for i in range(500)])
        connection.commit()
    sql = "SELECT COUNT(*) FROM nums a, nums b, nums c"
    result = execute_sql(sql, db_path, timeout=0)
    assert result.success is False
    assert result.error == "timeout"
