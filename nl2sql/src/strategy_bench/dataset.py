"""Dataset loading for strategy-bench cases."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias

import yaml


Scalar: TypeAlias = str | int | float | bool | None
ExpectedResultRow: TypeAlias = list[Scalar] | dict[str, Scalar]
ExpectedResult: TypeAlias = list[ExpectedResultRow]


@dataclass(slots=True)
class TestCase:
    """Normalized strategy-bench test case."""

    id: str
    natural_language_query: str
    expected_sql: str | None = None
    expected_result: ExpectedResult | None = None
    db_target: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


TestCase.__test__ = False


class DatasetLoader:
    """Load strategy-bench datasets from JSON/JSONL/YAML."""

    SUPPORTED_SUFFIXES = {".json", ".jsonl", ".yaml", ".yml"}

    def load(self, path: Path) -> list[TestCase]:
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported dataset format: {path.suffix}")
        rows = self._load_rows(path, suffix)
        return [self._normalize_case(row, index) for index, row in enumerate(rows)]

    def iter_cases(self, path: Path):
        for case in self.load(path):
            yield case

    def _load_rows(self, path: Path, suffix: str) -> list[dict[str, Any]]:
        if suffix == ".jsonl":
            rows: list[dict[str, Any]] = []
            with path.open("r", encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        payload = json.loads(stripped)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Malformed JSONL at {path}:{line_no}") from exc
                    if not isinstance(payload, dict):
                        raise ValueError(f"Expected object at {path}:{line_no}")
                    rows.append(payload)
            return rows

        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) if suffix in {".yaml", ".yml"} else json.load(handle)

        if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
            payload = payload["cases"]
        if not isinstance(payload, list):
            raise ValueError(f"Expected a top-level list or {{cases: [...]}} in {path}")
        rows = []
        for index, row in enumerate(payload):
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at index {index} in {path}")
            rows.append(row)
        return rows

    def _normalize_case(self, row: dict[str, Any], index: int) -> TestCase:
        case_id = self._required_str(row, "id", index)
        query = self._required_str(row, "natural_language_query", index)
        expected_sql = self._optional_str(row.get("expected_sql"))
        expected_result = self._normalize_expected_result(row.get("expected_result"), index)
        if expected_sql is None and expected_result is None:
            raise ValueError(
                f"Case {case_id!r} must define at least one of expected_sql or expected_result"
            )
        metadata = row.get("metadata")
        if metadata is None:
            metadata_dict: dict[str, Any] = {}
        elif isinstance(metadata, dict):
            metadata_dict = dict(metadata)
        else:
            raise ValueError(f"Case {case_id!r} metadata must be an object if provided")
        return TestCase(
            id=case_id,
            natural_language_query=query,
            expected_sql=expected_sql,
            expected_result=expected_result,
            db_target=self._optional_str(row.get("db_target")),
            metadata=metadata_dict,
        )

    @staticmethod
    def _required_str(row: dict[str, Any], key: str, index: int) -> str:
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Case at index {index} has invalid {key}")
        return value.strip()

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"Expected string value, got {type(value).__name__}")
        stripped = value.strip()
        return stripped or None

    @staticmethod
    def _normalize_expected_result(value: Any, index: int) -> ExpectedResult | None:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError(f"Case at index {index} expected_result must be a list")
        normalized: ExpectedResult = []
        for row in value:
            if isinstance(row, list):
                normalized.append(list(row))
                continue
            if isinstance(row, dict):
                normalized.append(dict(row))
                continue
            raise ValueError(
                f"Case at index {index} expected_result rows must be lists or objects"
            )
        return normalized
