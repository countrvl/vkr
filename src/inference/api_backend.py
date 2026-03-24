"""OpenAI-compatible API backend for frontier models."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, BadRequestError, RateLimitError

from src.inference.base import GenerationResult, InferenceBackend


LOGGER = logging.getLogger(__name__)
RETRY_DELAYS_SECONDS = (1.0, 2.0, 4.0)


class APIBackend(InferenceBackend):
    """Inference backend for OpenAI-compatible chat completion APIs."""

    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str,
        model_name: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the API backend.

        Args:
            model_id: Provider model identifier.
            base_url: OpenAI-compatible base URL.
            api_key: Provider API key.
            model_name: Human-readable model name.
            parameters: Extra request parameters.
        """
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_id = model_id
        self.model_name = model_name or model_id
        self.parameters = parameters or {}

    async def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> list[GenerationResult]:
        """Generate one or more SQL candidates."""
        try:
            return await self._generate_with_retry(
                prompt=prompt,
                n=n,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except BadRequestError:
            if n <= 1:
                raise
            LOGGER.warning(
                "Model %s rejected n=%s; falling back to %s separate requests.",
                self.model_id,
                n,
                n,
            )
            results: list[GenerationResult] = []
            for _ in range(n):
                results.extend(
                    await self._generate_with_retry(
                        prompt=prompt,
                        n=1,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                )
            return results

    async def _generate_with_retry(
        self,
        *,
        prompt: str,
        n: int,
        temperature: float,
        max_tokens: int,
    ) -> list[GenerationResult]:
        """Retry transient API errors with exponential backoff."""
        for attempt in range(1, len(RETRY_DELAYS_SECONDS) + 2):
            try:
                return await self._generate_native(
                    prompt=prompt,
                    n=n,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except (APIConnectionError, APITimeoutError, RateLimitError) as exc:
                if attempt > len(RETRY_DELAYS_SECONDS):
                    raise
                delay = RETRY_DELAYS_SECONDS[attempt - 1]
                LOGGER.warning(
                    "API request failed for %s (attempt %s/%s): %s. Retrying in %.1fs.",
                    self.model_id,
                    attempt,
                    len(RETRY_DELAYS_SECONDS) + 1,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
        raise RuntimeError("API generation failed without a captured error.")

    async def _generate_native(
        self,
        *,
        prompt: str,
        n: int,
        temperature: float,
        max_tokens: int,
    ) -> list[GenerationResult]:
        """Perform a single API request without retry logic."""
        request_params = dict(self.parameters)
        request_params.pop("max_tokens", None)

        started_at = time.perf_counter()
        response = await self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            **request_params,
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
                    sql=self.extract_sql(content or ""),
                    raw_response=content or "",
                    tokens_input=per_choice_prompt,
                    tokens_output=per_choice_completion,
                    latency_ms=latency_ms,
                    model_name=self.model_name,
                    metadata={"backend": "api"},
                )
            )
        return results


ApiInferenceBackend = APIBackend
