"""Shared inference abstractions."""

from __future__ import annotations

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
    ) -> list[GenerationResult]:
        """Generate SQL candidates."""

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

        cleaned = raw.strip()
        lowered = cleaned.lower()
        if any(pattern in lowered for pattern in REFUSAL_PATTERNS):
            return ""

        sql_fence = re.search(r"```sql\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if sql_fence:
            cleaned = sql_fence.group(1)
        else:
            generic_fence = re.search(r"```\s*(.*?)```", cleaned, flags=re.DOTALL)
            if generic_fence:
                cleaned = generic_fence.group(1)

        cleaned = cleaned.strip()
        if not cleaned:
            return ""

        cleaned = cleaned.rstrip().rstrip(";").strip()
        return cleaned if cleaned else ""


def extract_sql(raw: str) -> str:
    """Backward-compatible wrapper around `InferenceBackend.extract_sql()`."""
    return InferenceBackend.extract_sql(raw)
