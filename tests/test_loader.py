import sqlite3
from pathlib import Path

import pytest

from src.data.loader import DataSample, load_benchmark
from src.data.schema import serialize_schema


def test_data_sample_fields() -> None:
    sample = DataSample(
        id="spider_dev_0",
        benchmark="spider",
        question="How many users?",
        gold_sql="SELECT COUNT(*) FROM users;",
        db_id="demo",
        db_path=Path("demo.sqlite"),
        schema="CREATE TABLE users (id INTEGER);",
        difficulty="easy",
    )
    assert sample.id == "spider_dev_0"
    assert sample.db_path.name == "demo.sqlite"


def test_load_benchmark_rejects_unknown_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_benchmark("unknown", tmp_path)


def test_load_benchmark_missing_dev_file_raises_file_not_found(tmp_path: Path) -> None:
    (tmp_path / "spider").mkdir()
    with pytest.raises(FileNotFoundError):
        load_benchmark("spider", tmp_path)


def test_serialize_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "test.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        connection.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL)")
        connection.commit()

    schema = serialize_schema(db_path)
    assert "CREATE TABLE" in schema
    assert "users" in schema
    assert "orders" in schema
    # Two tables → two CREATE TABLE blocks separated by a blank line.
    assert schema.count("CREATE TABLE") == 2
