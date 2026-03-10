"""Spider dataset loader using a normalized JSONL contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_KEYS = ("question", "schema", "gold_sql")


class SpiderLoader:
    """Load Spider samples from a JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[dict[str, Any]]:
        """Load and validate Spider records.

        Returns:
            List of records with required keys question, schema, gold_sql
            and optional db_path.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Spider dataset file not found: {self.path}")

        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                self._validate_row(row, idx)
                records.append(
                    {
                        "question": str(row["question"]),
                        "schema": str(row["schema"]),
                        "gold_sql": str(row["gold_sql"]),
                        "db_path": str(row.get("db_path", "")),
                    }
                )
        return records

    @staticmethod
    def _validate_row(row: dict[str, Any], idx: int) -> None:
        missing = [key for key in REQUIRED_KEYS if key not in row]
        if missing:
            raise ValueError(f"Spider record {idx} is missing keys: {missing}")
