"""OpenAI-compatible transport shared across domains."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, TypeVar

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, BadRequestError, RateLimitError


LOGGER = logging.getLogger(__name__)
RETRY_DELAYS_SECONDS = (1.0, 2.0, 4.0)
ResultT = TypeVar("ResultT")


class OpenAIChatTransport:
    """Reusable OpenAI-compatible chat completions transport."""

    def __init__(
        self,
        *,
        model_id: str,
        base_url: str,
        api_key: str,
        model_name: str,
        parameters: dict[str, Any] | None = None,
        pricing: dict[str, Any] | None = None,
        response_format: Any | None = None,
        extractor: Callable[[str], str],
        result_factory: Callable[..., ResultT],
        disable_response_format_on_bad_request: bool = False,
    ) -> None:
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_id = model_id
        self.model_name = model_name
        self.parameters = parameters or {}
        self.pricing = self._normalize_pricing(pricing)
        self.response_format = response_format
        self.extractor = extractor
        self.result_factory = result_factory
        self.disable_response_format_on_bad_request = disable_response_format_on_bad_request
        self._structured_output_enabled = response_format is not None
        self._warned_rejected_n_values: set[int] = set()
        self._warned_partial_choice_counts: set[tuple[int, int]] = set()

    @staticmethod
    def _normalize_pricing(pricing: dict[str, Any] | None) -> dict[str, float] | None:
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
        *,
        prompt: str,
        n: int,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        top_p: float | None,
    ) -> list[ResultT]:
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
            results: list[ResultT] = []
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
        results: list[ResultT],
        requested_n: int,
        prompt: str,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        top_p: float | None,
    ) -> list[ResultT]:
        if requested_n <= 1 or len(results) >= requested_n:
            return results[:requested_n]

        missing = requested_n - len(results)
        partial_choice_key = (requested_n, len(results))
        if partial_choice_key not in self._warned_partial_choice_counts:
            LOGGER.debug(
                "Model %s returned only %s/%s choices; fetching %s more.",
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
    ) -> list[ResultT]:
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
        raise AssertionError("unreachable: retry loop should always return or raise")

    async def _generate_native(
        self,
        *,
        prompt: str,
        n: int,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        top_p: float | None,
    ) -> list[ResultT]:
        request_params = dict(self.parameters)
        request_params.pop("max_tokens", None)
        if top_p is not None:
            request_params["top_p"] = top_p
        if seed is not None:
            request_params["seed"] = seed
        response_format = request_params.pop("response_format", None)
        if response_format is None and self._structured_output_enabled:
            response_format = self.response_format

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
            if (
                response_format is None
                or not self.disable_response_format_on_bad_request
                or not self._should_disable_structured_output(exc)
            ):
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
        choices = getattr(response, "choices", []) or []
        n_choices = max(len(choices), 1)
        per_choice_out = [completion_tokens // n_choices] * n_choices
        if per_choice_out:
            per_choice_out[-1] += completion_tokens - sum(per_choice_out)

        results: list[ResultT] = []
        for idx, choice in enumerate(choices):
            content = choice.message.content if choice.message else ""
            cost_usd = None
            if self.pricing:
                cost_usd = (
                    (prompt_tokens / 1_000_000.0) * float(self.pricing.get("input_per_mtok", 0.0))
                    + (per_choice_out[idx] / 1_000_000.0) * float(self.pricing.get("output_per_mtok", 0.0))
                )
            metadata = {
                "backend": "api",
                "cost_usd": cost_usd,
                **({"pricing": dict(self.pricing)} if self.pricing else {}),
            }
            results.append(
                self.result_factory(
                    extracted=self.extractor(content or ""),
                    raw_response=content or "",
                    tokens_input=prompt_tokens,
                    tokens_output=per_choice_out[idx],
                    latency_ms=latency_ms,
                    model_name=self.model_name,
                    metadata=metadata,
                )
            )
        return results

    @staticmethod
    def _should_disable_structured_output(exc: BadRequestError) -> bool:
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
