"""OpenAI-compatible API backend for frontier models."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, BadRequestError, RateLimitError

from nl2sql.src.inference.base import GenerationResult, InferenceBackend


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
        pricing: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the API backend.

        Args:
            model_id: Provider model identifier.
            base_url: OpenAI-compatible base URL.
            api_key: Provider API key.
            model_name: Human-readable model name.
            parameters: Extra request parameters.
            pricing: Optional per-million token pricing from config.
        """
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_id = model_id
        self.model_name = model_name or model_id
        self.parameters = parameters or {}
        self.pricing = self._normalize_pricing(pricing)
        self._structured_output_enabled = True
        self._warned_rejected_n_values: set[int] = set()
        self._warned_partial_choice_counts: set[tuple[int, int]] = set()

    @staticmethod
    def _normalize_pricing(pricing: dict[str, Any] | None) -> dict[str, float] | None:
        """Convert config pricing keys into evaluation metadata format."""
        if not pricing:
            return None

        normalized: dict[str, float] = {}
        key_map = {
            "input_per_1m": "input_per_mtok",
            "output_per_1m": "output_per_mtok",
            "cache_hit_per_1m": "cache_hit_per_mtok",
        }
        for config_key, metadata_key in key_map.items():
            value = pricing.get(config_key)
            if value is not None:
                normalized[metadata_key] = float(value)
        return normalized or None

    async def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        top_p: float | None = None,
    ) -> list[GenerationResult]:
        """Generate one or more SQL candidates."""
        try:
            results = await self._generate_with_retry(
                prompt=prompt,
                n=n,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                top_p=top_p,
            )
            return await self._ensure_result_count(
                results=results,
                requested_n=n,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                top_p=top_p,
            )
        except BadRequestError:
            if n <= 1:
                raise
            if n not in self._warned_rejected_n_values:
                LOGGER.debug(
                    "Model %s rejected n=%s; falling back to %s separate requests.",
                    self.model_id,
                    n,
                    n,
                )
                self._warned_rejected_n_values.add(n)
            results: list[GenerationResult] = []
            for _ in range(n):
                results.extend(
                    await self._generate_with_retry(
                        prompt=prompt,
                        n=1,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        seed=seed,
                        top_p=top_p,
                    )
                )
            return results

    async def _ensure_result_count(
        self,
        *,
        results: list[GenerationResult],
        requested_n: int,
        prompt: str,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        top_p: float | None,
    ) -> list[GenerationResult]:
        """Top up missing candidates when a provider ignores or truncates ``n``."""
        if requested_n <= 1 or len(results) >= requested_n:
            return results[:requested_n]

        missing = requested_n - len(results)
        partial_choice_key = (requested_n, len(results))
        if partial_choice_key not in self._warned_partial_choice_counts:
            LOGGER.debug(
                "Model %s returned only %s/%s choices; fetching %s additional single-choice requests.",
                self.model_id,
                len(results),
                requested_n,
                missing,
            )
            self._warned_partial_choice_counts.add(partial_choice_key)
        topped_up = list(results)
        for _ in range(missing):
            topped_up.extend(
                await self._generate_with_retry(
                    prompt=prompt,
                    n=1,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    top_p=top_p,
                )
            )
        return topped_up[:requested_n]

    async def _generate_with_retry(
        self,
        *,
        prompt: str,
        n: int,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        top_p: float | None,
    ) -> list[GenerationResult]:
        """Retry transient API errors with exponential backoff."""
        for attempt in range(1, len(RETRY_DELAYS_SECONDS) + 2):
            try:
                return await self._generate_native(
                    prompt=prompt,
                    n=n,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    top_p=top_p,
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
        seed: int | None,
        top_p: float | None,
    ) -> list[GenerationResult]:
        """Perform a single API request without retry logic."""
        request_params = dict(self.parameters)
        request_params.pop("max_tokens", None)
        if top_p is not None:
            request_params["top_p"] = top_p
        if seed is not None:
            request_params["seed"] = seed
        response_format = request_params.pop("response_format", None)
        if response_format is None and self._structured_output_enabled:
            response_format = {"type": "json_object"}

        started_at = time.perf_counter()
        create_kwargs = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "n": n,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **request_params,
        }
        if response_format is not None:
            create_kwargs["response_format"] = response_format

        try:
            response = await self.client.chat.completions.create(**create_kwargs)
        except BadRequestError as exc:
            if response_format is None or not self._should_disable_structured_output(exc):
                raise
            LOGGER.debug(
                "Structured output is not supported by %s; retrying without response_format.",
                self.model_id,
            )
            self._structured_output_enabled = False
            fallback_kwargs = dict(create_kwargs)
            fallback_kwargs.pop("response_format", None)
            response = await self.client.chat.completions.create(**fallback_kwargs)
        latency_ms = (time.perf_counter() - started_at) * 1000.0

        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        results: list[GenerationResult] = []
        choices = getattr(response, "choices", []) or []
        n_choices = max(len(choices), 1)
        # Prompt tokens are shared across all choices (same input processed once).
        # Completion tokens are distributed per-choice; the remainder is added
        # to the last choice so the total is always preserved exactly.
        per_choice_out = [completion_tokens // n_choices] * n_choices
        if per_choice_out:
            per_choice_out[-1] += completion_tokens - sum(per_choice_out)
        for idx, choice in enumerate(choices):
            content = choice.message.content if choice.message else ""
            results.append(
                GenerationResult(
                    sql=self.extract_sql(content or ""),
                    raw_response=content or "",
                    tokens_input=prompt_tokens,
                    tokens_output=per_choice_out[idx],
                    latency_ms=latency_ms,
                    model_name=self.model_name,
                    metadata={
                        "backend": "api",
                        **({"pricing": dict(self.pricing)} if self.pricing else {}),
                    },
                )
            )
        return results

    @staticmethod
    def _should_disable_structured_output(exc: BadRequestError) -> bool:
        """Return True when the provider rejects structured output options."""
        message = str(exc).lower()
        return any(
            token in message
            for token in (
                "response_format",
                "json_object",
                "json schema",
                "structured output",
                "not supported",
                "unsupported",
            )
        )


ApiInferenceBackend = APIBackend
