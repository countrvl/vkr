"""OpenAI-compatible API model adapter."""

from __future__ import annotations

import os
from typing import Any

import requests

from src.models.base_model import BaseModel


class APIModel(BaseModel):
    """Adapter for OpenAI-style /v1/chat/completions endpoints."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "https://api.openai.com",
        timeout: int = 120,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set")

        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key

    def generate(self, prompt: str) -> str:
        text, _ = self.generate_with_metadata(prompt)
        return text

    def generate_with_metadata(self, prompt: str) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        choices = data.get("choices", [])
        content = ""
        if choices:
            content = str(choices[0].get("message", {}).get("content", "")).strip()

        metadata: dict[str, Any] = {"usage": data.get("usage", {})}
        return content, metadata
