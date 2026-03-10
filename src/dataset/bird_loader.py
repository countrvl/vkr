"""Лоадер BIRD в нормализованный JSONL-контракт."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_KEYS = ("question", "schema", "gold_sql")


class BirdLoader:
    """Загрузка примеров BIRD из JSONL-файла."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> list[dict[str, Any]]:
        """Загрузить и провалидировать записи BIRD.

        Возвращает:
            Список записей с обязательными полями question, schema, gold_sql
            и опциональным db_path.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Файл датасета BIRD не найден: {self.path}")

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
            raise ValueError(f"В записи BIRD #{idx} отсутствуют поля: {missing}")
