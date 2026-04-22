import json
from pathlib import Path

import pytest

from nl2sql.src.strategy_bench.dataset import DatasetLoader


def test_dataset_loader_supports_json_and_yaml(tmp_path: Path) -> None:
    payload = [
        {
            "id": "case-1",
            "natural_language_query": "Count users",
            "expected_sql": "SELECT COUNT(*) FROM users",
            "metadata": {"owner": "analytics"},
        }
    ]
    json_path = tmp_path / "cases.json"
    yaml_path = tmp_path / "cases.yaml"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    yaml_path.write_text(
        """
cases:
  - id: case-1
    natural_language_query: Count users
    expected_sql: SELECT COUNT(*) FROM users
    metadata:
      owner: analytics
""".strip(),
        encoding="utf-8",
    )

    loader = DatasetLoader()
    json_cases = loader.load(json_path)
    yaml_cases = loader.load(yaml_path)

    assert json_cases[0].id == "case-1"
    assert yaml_cases[0].metadata == {"owner": "analytics"}


def test_dataset_loader_rejects_missing_required_fields(tmp_path: Path) -> None:
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{"id": "case-1", "expected_sql": "SELECT 1"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="natural_language_query"):
        DatasetLoader().load(path)


def test_dataset_loader_requires_reference(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps({"id": "case-1", "natural_language_query": "Count users"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected_sql or expected_result"):
        DatasetLoader().load(path)
