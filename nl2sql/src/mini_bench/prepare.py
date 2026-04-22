"""Build and validate the local NL2SQL mini-benchmark assets."""

from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from nl2sql.src.strategy_bench.dataset import DatasetLoader, TestCase
from nl2sql.src.strategy_bench.executor import SQLiteExecutor
from nl2sql.src.strategy_bench.validation import ValidationModule


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data" / "nl2sql" / "mini_bench"
SNAPSHOT_SQL_PATH = DATA_ROOT / "snapshot.sql"
DATASET_PATH = DATA_ROOT / "cases.yaml"
DB_PATH = DATA_ROOT / "snapshot.sqlite"
EXPECTED_CASE_COUNT = 50


@dataclass(slots=True)
class DatasetValidationIssue:
    case_id: str
    message: str


def build_snapshot_db(
    *,
    output_path: Path = DB_PATH,
    sql_path: Path = SNAPSHOT_SQL_PATH,
    force: bool = False,
) -> Path:
    """Create the deterministic SQLite snapshot from the SQL seed file."""
    if output_path.exists() and not force:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    script = sql_path.read_text(encoding="utf-8")
    connection = sqlite3.connect(output_path)
    try:
        connection.executescript(script)
        connection.commit()
    finally:
        connection.close()
    return output_path


def load_cases(path: Path = DATASET_PATH) -> list[TestCase]:
    return DatasetLoader().load(path)


def validate_dataset(
    *,
    dataset_path: Path = DATASET_PATH,
    db_path: Path = DB_PATH,
    expected_case_count: int = EXPECTED_CASE_COUNT,
) -> tuple[list[TestCase], list[DatasetValidationIssue]]:
    """Validate dataset shape and executable references."""
    cases = load_cases(dataset_path)
    issues: list[DatasetValidationIssue] = []
    if len(cases) != expected_case_count:
        issues.append(
            DatasetValidationIssue(
                case_id="__dataset__",
                message=f"Expected {expected_case_count} cases, found {len(cases)}",
            )
        )

    category_counts = Counter(str(case.metadata.get("category", "")) for case in cases)
    difficulty_counts = Counter(str(case.metadata.get("difficulty", "")) for case in cases)
    missing_category = [key for key in category_counts if not key]
    missing_difficulty = [key for key in difficulty_counts if not key]
    if missing_category:
        issues.append(
            DatasetValidationIssue(
                case_id="__dataset__",
                message="Some cases are missing metadata.category",
            )
        )
    if missing_difficulty:
        issues.append(
            DatasetValidationIssue(
                case_id="__dataset__",
                message="Some cases are missing metadata.difficulty",
            )
        )

    executor = SQLiteExecutor(db_path)
    validator = ValidationModule()
    for case in cases:
        if case.expected_sql is None:
            issues.append(
                DatasetValidationIssue(
                    case_id=case.id,
                    message="Mini-benchmark cases must define expected_sql",
                )
            )
            continue
        validation = validator.validate(case, case.expected_sql, executor)
        if validation.invalid_reference:
            issues.append(
                DatasetValidationIssue(
                    case_id=case.id,
                    message="Reference SQL is invalid for the snapshot database",
                )
            )
        elif validation.execution_outcome is not None and not validation.execution_outcome.success:
            issues.append(
                DatasetValidationIssue(
                    case_id=case.id,
                    message=validation.execution_outcome.error_message or "Reference SQL failed",
                )
            )
        elif validation.issues:
            issues.append(
                DatasetValidationIssue(
                    case_id=case.id,
                    message=validation.issues[0].message,
                )
            )
    return cases, issues


def prepare_mini_bench(*, force: bool = False) -> tuple[Path, list[TestCase], list[DatasetValidationIssue]]:
    db_path = build_snapshot_db(force=force)
    cases, issues = validate_dataset(db_path=db_path)
    return db_path, cases, issues

