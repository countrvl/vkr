"""Адаптер модели Ollama."""

from __future__ import annotations

from typing import Any

import requests

from src.models.base_model import BaseModel


class OllamaModel(BaseModel):
    """Адаптер локальных моделей Ollama через /api/generate."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.options = options or {}

    def generate(self, prompt: str) -> str:
        text, _ = self.generate_with_metadata(prompt)
        return text

    def generate_with_metadata(self, prompt: str) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        if self.options:
            payload["options"] = self.options

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        generated = str(data.get("response", "")).strip()
        metadata: dict[str, Any] = {
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"),
            "total_duration": data.get("total_duration"),
        }
        return generated, metadata
