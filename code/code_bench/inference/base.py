"""Базовые абстракции инференса для генерации кода."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", flags=re.DOTALL)
_PYTHON_START_PATTERN = re.compile(r"(?m)^(from\s+\S+\s+import|import\s+\S+|def\s+\w+|class\s+\w+)")


@dataclass(slots=True)
class GenerationResult:
    """Нормализованный payload генерации кода."""

    code: str
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
        max_tokens: int = 768,
        seed: int | None = None,
        top_p: float | None = None,
    ) -> list[GenerationResult]:
        """Сгенерировать один или несколько кандидатов кода."""

    @staticmethod
    def extract_code(raw: str) -> str:
        """Извлечь Python-код из сырого ответа модели."""
        if not raw or not raw.strip():
            return ""

        cleaned = raw.strip()
        extracted_from_json = InferenceBackend._extract_code_from_json(cleaned)
        if extracted_from_json:
            cleaned = extracted_from_json

        code_fence = re.search(r"```python\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if code_fence:
            cleaned = code_fence.group(1)
        else:
            generic_fence = re.search(r"```\s*(.*?)```", cleaned, flags=re.DOTALL)
            if generic_fence:
                cleaned = generic_fence.group(1)

        cleaned = cleaned.strip()
        if not cleaned:
            return ""

        python_start = _PYTHON_START_PATTERN.search(cleaned)
        if python_start and python_start.start() > 0:
            cleaned = cleaned[python_start.start() :]

        return cleaned.strip()

    @staticmethod
    def _extract_code_from_json(raw: str) -> str:
        """Вернуть код из JSON-объекта с верхнеуровневым полем ``code``."""
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
            code = payload.get("code")
            if isinstance(code, str) and code.strip():
                return code.strip()
        return ""


def extract_code(raw: str) -> str:
    """Совместимая обертка вокруг `InferenceBackend.extract_code()`."""
    return InferenceBackend.extract_code(raw)
