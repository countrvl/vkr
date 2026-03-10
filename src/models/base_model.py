"""Base interface for all model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseModel(ABC):
    """Abstract model API for text generation."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a text completion for the prompt."""

    def generate_with_metadata(self, prompt: str) -> tuple[str, dict[str, Any]]:
        """Generate text and optional metadata.

        Default implementation wraps `generate` and returns empty metadata.
        """
        return self.generate(prompt), {}
