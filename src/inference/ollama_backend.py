"""Async Ollama backend for local NL2SQL models."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from src.inference.base import GenerationResult, InferenceBackend, SQL_RESPONSE_SCHEMA


LOGGER = logging.getLogger(__name__)
RETRY_DELAYS_SECONDS = (1.0, 2.0, 4.0)


class OllamaBackend(InferenceBackend):
    """Inference backend for Ollama's `/api/generate` endpoint."""

    def __init__(
        self,
        model_id: str,
        base_url: str = "http://localhost:11434",
        num_ctx: int = 4096,
        model_name: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the Ollama backend.

        Args:
            model_id: Ollama model tag.
            base_url: Ollama REST API base URL.
            num_ctx: Context window size.
            model_name: Human-readable model name.
            parameters: Additional Ollama options.
        """
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self.model_name = model_name or model_id
        self.parameters = parameters or {}
        self._structured_output_enabled = True

    async def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        top_p: float | None = None,
    ) -> list[GenerationResult]:
        """Generate SQL candidates sequentially through Ollama.

        ``top_p`` is forwarded to Ollama's ``options`` dict when provided.
        ``seed`` overrides the default reproducibility seed (42).
        """
        results: list[GenerationResult] = []
        async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
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
                    GenerationResult(
                        sql=self.extract_sql(raw_response),
                        raw_response=raw_response,
                        tokens_input=int(payload.get("prompt_eval_count", 0) or 0),
                        tokens_output=int(payload.get("eval_count", 0) or 0),
                        latency_ms=latency_ms,
                        model_name=self.model_name,
                        metadata={"backend": "ollama"},
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
        """Retry transient Ollama failures with exponential backoff."""
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
        raise RuntimeError("Ollama generation failed without a captured error.")

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
        """Perform a single Ollama request without retry logic."""
        options = dict(self.parameters)
        options.pop("max_tokens", None)
        options.pop("num_ctx", None)
        options.update(
            {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": self.num_ctx,
                "seed": seed if seed is not None else 42,
            }
        )
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
            payload["format"] = SQL_RESPONSE_SCHEMA

        try:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
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
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=payload,
            )
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
        """Return True when Ollama rejects the ``format`` schema parameter."""
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


OllamaInferenceBackend = OllamaBackend
