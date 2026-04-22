"""Model abstraction and adapters for strategy-bench."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from nl2sql.src.inference.base import InferenceBackend


@dataclass(slots=True)
class ModelResponse:
    """Normalized SQL generation result."""

    sql: str
    raw_response: str
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelInterface(ABC):
    """Synchronous model interface for strategy execution."""

    @abstractmethod
    def generate_sql(self, prompt: str) -> ModelResponse:
        """Generate one SQL query for the provided prompt."""


class BackendModelAdapter(ModelInterface):
    """Wrap an existing async `InferenceBackend` into the local interface."""

    def __init__(
        self,
        backend: InferenceBackend,
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        top_p: float | None = None,
    ) -> None:
        self._backend = backend
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._seed = seed
        self._top_p = top_p

    def generate_sql(self, prompt: str) -> ModelResponse:
        generations = asyncio.run(
            self._backend.generate(
                prompt=prompt,
                n=1,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                seed=self._seed,
                top_p=self._top_p,
            )
        )
        if not generations:
            raise RuntimeError("Inference backend returned no generations")
        generation = generations[0]
        return ModelResponse(
            sql=generation.sql,
            raw_response=generation.raw_response,
            latency_ms=generation.latency_ms,
            metadata=dict(generation.metadata),
        )
