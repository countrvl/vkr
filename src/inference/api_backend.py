"""OpenAI-compatible API backend for frontier models."""

from __future__ import annotations

import time
from typing import Any

from openai import AsyncOpenAI

from src.inference.base import GenerationResult, InferenceBackend, extract_sql


class ApiInferenceBackend(InferenceBackend):
    """Inference backend for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model_id: str,
        model_name: str,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model_id = model_id
        self._model_name = model_name
        self._parameters = parameters or {}

    async def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.0,
    ) -> list[GenerationResult]:
        try:
            return await self._generate_native(prompt=prompt, n=n, temperature=temperature)
        except Exception:
            if n <= 1:
                raise
            results: list[GenerationResult] = []
            for _ in range(n):
                results.extend(await self._generate_native(prompt=prompt, n=1, temperature=temperature))
            return results

    async def _generate_native(
        self,
        *,
        prompt: str,
        n: int,
        temperature: float,
    ) -> list[GenerationResult]:
        started_at = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self._model_id,
            messages=[{"role": "user", "content": prompt}],
            n=n,
            temperature=temperature,
            **self._parameters,
        )
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        results: list[GenerationResult] = []
        choices = getattr(response, "choices", []) or []
        per_choice_prompt = prompt_tokens // max(len(choices), 1)
        per_choice_completion = completion_tokens // max(len(choices), 1)
        for choice in choices:
            content = choice.message.content if choice.message else ""
            results.append(
                GenerationResult(
                    sql=extract_sql(content),
                    raw_response=content,
                    tokens_input=per_choice_prompt,
                    tokens_output=per_choice_completion,
                    latency_ms=latency_ms,
                    model_name=self._model_name,
                    metadata={"backend": "api"},
                )
            )
        return results
