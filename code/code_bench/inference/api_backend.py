"""OpenAI-compatible API-backend для моделей генерации кода."""

from __future__ import annotations

from typing import Any

from code_bench.inference.base import GenerationResult, InferenceBackend
from shared.inference.api_transport import OpenAIChatTransport


class APIBackend(InferenceBackend):
    """Backend инференса для OpenAI-compatible chat completion API."""

    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str,
        model_name: str | None = None,
        parameters: dict[str, Any] | None = None,
        pricing: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name or model_id
        self._transport = OpenAIChatTransport(
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            model_name=self.model_name,
            parameters=parameters,
            pricing=pricing,
            extractor=self.extract_code,
            result_factory=self._make_result,
        )

    @property
    def client(self):
        return self._transport.client

    @client.setter
    def client(self, value) -> None:
        self._transport.client = value

    @property
    def model_id(self) -> str:
        return self._transport.model_id

    async def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 768,
        seed: int | None = None,
        top_p: float | None = None,
    ) -> list[GenerationResult]:
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
        if requested_n <= 1 or len(results) >= requested_n:
            return results[:requested_n]

        missing = requested_n - len(results)
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
        return await self._transport._generate_with_retry(
            prompt=prompt,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            top_p=top_p,
        )

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
        return await self._transport._generate_native(
            prompt=prompt,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            top_p=top_p,
        )

    @staticmethod
    def _make_result(
        *,
        extracted: str,
        raw_response: str,
        tokens_input: int,
        tokens_output: int,
        latency_ms: float,
        model_name: str,
        metadata: dict[str, Any],
    ) -> GenerationResult:
        return GenerationResult(
            code=extracted,
            raw_response=raw_response,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=latency_ms,
            model_name=model_name,
            metadata=metadata,
        )
