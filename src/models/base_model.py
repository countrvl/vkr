"""Базовый интерфейс для всех адаптеров моделей."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """Абстрактный API модели для генерации текста."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Сгенерировать текстовый ответ на промпт."""

    def generate_with_metadata(self, prompt: str) -> tuple[str, dict[str, Any]]:
        """Сгенерировать текст и опциональные метаданные.

        По умолчанию оборачивает `generate` и возвращает пустые метаданные.
        """
        return self.generate(prompt), {}
