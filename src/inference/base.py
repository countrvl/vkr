"""Shared inference abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


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


class InferenceBackend(ABC):
    """Interface for async inference backends."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.0,
    ) -> list[GenerationResult]:
        """Generate SQL candidates."""


def extract_sql(text: str) -> str:
    """Best-effort SQL extraction from model output."""
    cleaned = text.strip()
    if "```" in cleaned:
        blocks = [block.strip() for block in cleaned.split("```") if block.strip()]
        for block in blocks:
            if block.lower().startswith("sql"):
                return block[3:].strip()
        return blocks[0]
    return cleaned
