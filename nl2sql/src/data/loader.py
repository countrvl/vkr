"""Unified benchmark loaders for Spider and BIRD."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nl2sql.src.data.schema import serialize_schema

SUPPORTED_BENCHMARKS = {"spider", "bird"}
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DataSample:
    """Normalized benchmark sample."""

    id: str
    benchmark: str
    question: str
    gold_sql: str
    db_id: str
    db_path: Path
    schema: str
    difficulty: str | None
    evidence: str | None = None


def load_spider(data_dir: Path) -> list[DataSample]:
    """Load Spider dev samples into the unified format.

    Args:
        data_dir: Root benchmark directory.

    Returns:
        Loaded Spider samples with serialized schemas.
    """
    spider_dir = data_dir / "spider"
    dev_path = spider_dir / "dev.json"
    if not dev_path.exists():
        raise FileNotFoundError(f"Spider dev file not found: {dev_path}")

    entries = _load_json_array(dev_path)
    samples: list[DataSample] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            LOGGER.warning("Skipping Spider row %s: expected object, got %s", index, type(entry).__name__)
            continue

        question = _get_required_str(entry, "question", f"Spider row {index}")
        gold_sql = _get_required_str(entry, "query", f"Spider row {index}")
        db_id = _get_required_str(entry, "db_id", f"Spider row {index}")
        if question is None or gold_sql is None or db_id is None:
            continue

        db_path = spider_dir / "database" / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            LOGGER.warning("Skipping Spider row %s: database file not found at %s", index, db_path)
            continue

        samples.append(
            DataSample(
                id=f"spider_{index}",
                benchmark="spider",
                question=question,
                gold_sql=gold_sql,
                db_id=db_id,
                db_path=db_path,
                schema=serialize_schema(db_path),
                difficulty=None,
            )
        )
    skipped = len(entries) - len(samples)
    log = LOGGER.warning if skipped else LOGGER.info
    log("Spider: loaded %d/%d samples (%d skipped).", len(samples), len(entries), skipped)
    return samples


def load_bird(data_dir: Path) -> list[DataSample]:
    """Load BIRD dev samples into the unified format.

    Args:
        data_dir: Root benchmark directory.

    Returns:
        Loaded BIRD samples with serialized schemas.
    """
    bird_dir = data_dir / "bird"
    dev_path = bird_dir / "dev.json"
    if not dev_path.exists():
        raise FileNotFoundError(f"BIRD dev file not found: {dev_path}")
    databases_dir = _resolve_bird_databases_dir(bird_dir)

    entries = _load_json_array(dev_path)
    samples: list[DataSample] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            LOGGER.warning("Skipping BIRD row %s: expected object, got %s", index, type(entry).__name__)
            continue

        question_id = entry.get("question_id", index)
        question = _get_required_str(entry, "question", f"BIRD row {index}")
        gold_sql = _get_required_str(entry, "SQL", f"BIRD row {index}")
        db_id = _get_required_str(entry, "db_id", f"BIRD row {index}")
        if question is None or gold_sql is None or db_id is None:
            continue

        db_path = databases_dir / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            LOGGER.warning("Skipping BIRD row %s: database file not found at %s", index, db_path)
            continue

        samples.append(
            DataSample(
                id=f"bird_{question_id}",
                benchmark="bird",
                question=question,
                gold_sql=gold_sql,
                db_id=db_id,
                db_path=db_path,
                schema=serialize_schema(db_path),
                difficulty=_optional_str(entry.get("difficulty")),
                evidence=_optional_str(entry.get("evidence")),
            )
        )
    skipped = len(entries) - len(samples)
    log = LOGGER.warning if skipped else LOGGER.info
    log("BIRD: loaded %d/%d samples (%d skipped).", len(samples), len(entries), skipped)
    return samples


def load_benchmark(name: str, data_dir: Path) -> list[DataSample]:
    """Load Spider or BIRD into a unified sample list.

    Args:
        name: Benchmark name (`spider` or `bird`).
        data_dir: Root benchmark directory.

    Returns:
        Normalized benchmark samples.

    Raises:
        ValueError: If the benchmark is not supported.
    """
    benchmark = name.lower()
    if benchmark not in SUPPORTED_BENCHMARKS:
        raise ValueError(f"Unsupported benchmark: {name}")

    if benchmark == "spider":
        return load_spider(data_dir)
    return load_bird(data_dir)


def _load_json_array(path: Path) -> list[Any]:
    """Load a JSON array from disk."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON array in {path}, got {type(payload).__name__}")
    return payload


def _get_required_str(entry: dict[str, Any], key: str, row_name: str) -> str | None:
    """Extract a required non-empty string field."""
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        LOGGER.warning("Skipping %s: missing or invalid %s", row_name, key)
        return None
    return value


def _optional_str(value: Any) -> str | None:
    """Normalize an optional string field."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _resolve_bird_databases_dir(bird_dir: Path) -> Path:
    """Return the canonical BIRD databases directory, handling nested layouts."""
    databases_dir = bird_dir / "dev_databases"
    if not databases_dir.exists():
        raise FileNotFoundError(f"BIRD dev_databases directory not found: {databases_dir}")

    nested_dir = databases_dir / "dev_databases"
    if nested_dir.exists() and any(path.is_file() for path in nested_dir.rglob("*.sqlite")):
        return nested_dir
    if any(path.is_file() for path in databases_dir.rglob("*.sqlite")):
        return databases_dir
    raise FileNotFoundError(
        f"BIRD dataset is incomplete: no SQLite databases found under {databases_dir}. "
        "Re-run scripts/01_download_data.py --benchmark bird."
    )
