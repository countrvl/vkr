from pathlib import Path

import pytest

from src.data.loader import DataSample, load_benchmark


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


def test_load_benchmark_scaffold_raises_not_implemented(tmp_path: Path) -> None:
    (tmp_path / "spider").mkdir()
    with pytest.raises(NotImplementedError):
        load_benchmark("spider", tmp_path)
