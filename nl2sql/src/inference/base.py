"""Базовые абстракции инференса для NL2SQL."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


REFUSAL_PATTERNS = (
    "i cannot generate sql",
    "i can't generate sql",
    "cannot generate sql",
    "unable to generate sql",
)

_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", flags=re.DOTALL)
SQL_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sql": {"type": "string"},
    },
    "required": ["sql"],
    "additionalProperties": False,
}


@dataclass(slots=True)
class GenerationResult:
    """Нормализованный payload генерации."""

    sql: str
    raw_response: str
    tokens_input: int
    tokens_output: int
    latency_ms: float
    model_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class InferenceBackend(ABC):
    """Интерфейс async-backend-ов инференса."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        top_p: float | None = None,
    ) -> list[GenerationResult]:
        """Сгенерировать SQL-кандидаты.

        Аргументы:
            prompt: уже собранный текст prompt-а.
            n: число независимых генераций.
            temperature: температура сэмплирования (`0` = greedy).
            max_tokens: максимальное число выходных токенов.
            seed: необязательный seed для воспроизводимости.
            top_p: необязательный порог nucleus sampling.
        """

    @staticmethod
    def extract_sql(raw: str) -> str:
        """Извлечь SQL из сырого ответа модели.

        Возвращает очищенный SQL или пустую строку.
        """
        if not raw or not raw.strip():
            return ""

        cleaned = normalize_sql_text(raw)
        lowered = cleaned.lower()
        if any(pattern in lowered for pattern in REFUSAL_PATTERNS):
            return ""

        extracted_from_json = InferenceBackend._extract_sql_from_json(cleaned)
        if extracted_from_json:
            return extracted_from_json

        sql_fence = re.search(r"```sql\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if sql_fence:
            cleaned = sql_fence.group(1)
        else:
            generic_fence = re.search(r"```\s*(.*?)```", cleaned, flags=re.DOTALL)
            if generic_fence:
                cleaned = generic_fence.group(1)

        cleaned = normalize_sql_text(cleaned)
        if not cleaned:
            return ""

        return cleaned if cleaned else ""

    @staticmethod
    def _extract_sql_from_json(raw: str) -> str:
        """Вернуть SQL из JSON-ответа с верхнеуровневым полем ``sql``."""
        candidates = [raw]
        json_match = _JSON_OBJECT_PATTERN.search(raw)
        if json_match is not None and json_match.group(0) != raw:
            candidates.append(json_match.group(0))

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            sql = payload.get("sql")
            if not isinstance(sql, str):
                continue
            cleaned = normalize_sql_text(sql)
            if cleaned:
                return cleaned
        return ""


def extract_sql(raw: str) -> str:
    """Совместимая обертка вокруг `InferenceBackend.extract_sql()`."""
    return InferenceBackend.extract_sql(raw)


def normalize_sql_text(sql: str) -> str:
    """Нормализовать SQL модели перед сохранением или выполнением."""
    if not sql:
        return ""
    cleaned = sql.replace("\\n", "\n").replace("\\t", "\t").strip()
    cleaned = cleaned.rstrip().rstrip(";").strip()
    return cleaned
