"""Shared inference abstractions."""

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
    """Normalized generation payload."""

    sql: str
    raw_response: str
    tokens_input: int
    tokens_output: int
    latency_ms: float
    model_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class InferenceBackend(ABC):
    """Interface for async inference backends."""

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
        """Generate SQL candidates.

        Args:
            prompt: Rendered prompt text.
            n: Number of independent completions to generate.
            temperature: Sampling temperature (0 = greedy).
            max_tokens: Maximum output tokens.
            seed: Optional random seed for reproducibility.
            top_p: Optional nucleus sampling threshold.
        """

    @staticmethod
    def extract_sql(raw: str) -> str:
        """Extract SQL from a raw model response.

        Args:
            raw: Full model output.

        Returns:
            Cleaned SQL string or an empty string.
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
        """Return SQL from a JSON response with a top-level ``sql`` field."""
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
    """Backward-compatible wrapper around `InferenceBackend.extract_sql()`."""
    return InferenceBackend.extract_sql(raw)


def normalize_sql_text(sql: str) -> str:
    """Normalize model-output SQL before persistence or execution."""
    if not sql:
        return ""
    cleaned = sql.replace("\\n", "\n").replace("\\t", "\t").strip()
    cleaned = cleaned.rstrip().rstrip(";").strip()
    return cleaned
