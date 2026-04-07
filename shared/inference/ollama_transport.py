"""Ollama `/api/generate` transport shared across domains."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, TypeVar

import httpx


LOGGER = logging.getLogger(__name__)
RETRY_DELAYS_SECONDS = (1.0, 2.0, 4.0)
ResultT = TypeVar("ResultT")


class OllamaGenerateTransport:
    """Reusable Ollama generate transport."""

    def __init__(
        self,
        *,
        model_id: str,
        base_url: str,
        num_ctx: int,
        model_name: str,
        parameters: dict[str, Any] | None = None,
        extractor: Callable[[str], str],
        result_factory: Callable[..., ResultT],
        format_schema: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self.model_name = model_name
        self.parameters = dict(parameters or {})
        self.request_timeout = float(self.parameters.pop("request_timeout", 120.0))
        self.extractor = extractor
        self.result_factory = result_factory
        self.format_schema = format_schema
        self._structured_output_enabled = format_schema is not None

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
        results: list[ResultT] = []
        async with httpx.AsyncClient(timeout=self.request_timeout, trust_env=False) as client:
            for _ in range(n):
                payload, latency_ms = await self._generate_with_retry(
                    client=client,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    top_p=top_p,
                )
                raw_response = str(payload.get("response", ""))
                results.append(
                    self.result_factory(
                        extracted=self.extractor(raw_response),
                        raw_response=raw_response,
                        tokens_input=int(payload.get("prompt_eval_count", 0) or 0),
                        tokens_output=int(payload.get("eval_count", 0) or 0),
                        latency_ms=latency_ms,
                        model_name=self.model_name,
                        metadata={"backend": "ollama", "cost_usd": 0.0},
                    )
                )
        return results

    async def _generate_with_retry(
        self,
        *,
        client: httpx.AsyncClient,
        prompt: str,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        top_p: float | None,
    ) -> tuple[dict[str, Any], float]:
        for attempt in range(1, len(RETRY_DELAYS_SECONDS) + 2):
            try:
                return await self._generate_once(
                    client=client,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    top_p=top_p,
                )
            except httpx.ConnectError as exc:
                if attempt > len(RETRY_DELAYS_SECONDS):
                    raise RuntimeError("Ollama not running. Start with: ollama serve") from exc
                delay = RETRY_DELAYS_SECONDS[attempt - 1]
                LOGGER.warning(
                    "Ollama connection failed (attempt %s/%s): %s. Retrying in %.1fs.",
                    attempt,
                    len(RETRY_DELAYS_SECONDS) + 1,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
            except (httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                if attempt > len(RETRY_DELAYS_SECONDS):
                    raise
                delay = RETRY_DELAYS_SECONDS[attempt - 1]
                LOGGER.warning(
                    "Ollama request failed for %s (attempt %s/%s): %s. Retrying in %.1fs.",
                    self.model_id,
                    attempt,
                    len(RETRY_DELAYS_SECONDS) + 1,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
        raise AssertionError("unreachable: retry loop should always return or raise")

    async def _generate_once(
        self,
        *,
        client: httpx.AsyncClient,
        prompt: str,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        top_p: float | None,
    ) -> tuple[dict[str, Any], float]:
        options = dict(self.parameters)
        options.pop("max_tokens", None)
        options.pop("num_ctx", None)
        options.update(
            {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
            }
        )
        if seed is not None:
            options["seed"] = seed
        if top_p is not None:
            options["top_p"] = top_p

        started_at = time.perf_counter()
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if self._structured_output_enabled:
            payload["format"] = self.format_schema

        try:
            response = await client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if not payload.get("format") or not self._should_disable_structured_output(exc):
                raise
            LOGGER.warning(
                "Structured output is not supported by %s; retrying without format schema.",
                self.model_id,
            )
            self._structured_output_enabled = False
            payload.pop("format", None)
            response = await client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected Ollama response type: {type(payload).__name__}")

        wall_clock_ms = (time.perf_counter() - started_at) * 1000.0
        total_duration = payload.get("total_duration")
        if isinstance(total_duration, (int, float)):
            latency_ms = float(total_duration) / 1_000_000.0
        else:
            latency_ms = wall_clock_ms
        return payload, latency_ms

    @staticmethod
    def _should_disable_structured_output(exc: httpx.HTTPStatusError) -> bool:
        message = f"{exc} {getattr(exc.response, 'text', '')}".lower()
        return any(
            token in message
            for token in (
                "format",
                "schema",
                "json",
                "not supported",
                "unsupported",
                "invalid",
            )
        )
