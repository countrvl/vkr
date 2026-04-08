"""Anthropic backend для frontier-моделей в NL2SQL."""

from __future__ import annotations

from typing import Any

from nl2sql.src.inference.base import GenerationResult, InferenceBackend
from shared.inference.anthropic_transport import AnthropicMessagesTransport


class AnthropicBackend(InferenceBackend):
    """Backend инференса для Anthropic Messages API."""

    supports_batch = True

    def __init__(
        self,
        model_id: str,
        base_url: str,
        api_key: str,
        model_name: str | None = None,
        parameters: dict[str, Any] | None = None,
        pricing: dict[str, Any] | None = None,
        use_batch: bool = True,
    ) -> None:
        self.model_name = model_name or model_id
        self._transport = AnthropicMessagesTransport(
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            model_name=self.model_name,
            parameters=parameters,
            pricing=pricing,
            extractor=self.extract_sql,
            result_factory=self._make_result,
            use_batch=use_batch,
        )

    async def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        top_p: float | None = None,
    ) -> list[GenerationResult]:
        return await self._transport.generate(
            prompt=prompt,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            top_p=top_p,
        )

    async def generate_batch(
        self,
        prompts: list[str],
        *,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        top_p: float | None,
    ) -> list[list[GenerationResult] | Exception]:
        return await self._transport.generate_batch(
            prompts=prompts,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            top_p=top_p,
        )

    async def resume_batch(
        self,
        *,
        batch_id: str,
        prompts: list[str],
        temperature: float,
        max_tokens: int,
        seed: int | None,
        top_p: float | None,
    ) -> list[list[GenerationResult] | Exception]:
        return await self._transport.resume_batch(
            batch_id=batch_id,
            prompts=prompts,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            top_p=top_p,
        )

    def set_batch_status_callback(self, callback) -> None:
        self._transport.status_callback = callback

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
            sql=extracted,
            raw_response=raw_response,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=latency_ms,
            model_name=model_name,
            metadata=metadata,
        )
