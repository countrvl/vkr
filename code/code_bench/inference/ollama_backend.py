"""Async Ollama-backend для локальных моделей генерации кода."""

from __future__ import annotations

from typing import Any

from code_bench.inference.base import GenerationResult, InferenceBackend
from shared.inference.ollama_transport import OllamaGenerateTransport


class OllamaBackend(InferenceBackend):
    """Backend инференса для Ollama `/api/generate`."""

    def __init__(
        self,
        model_id: str,
        base_url: str = "http://localhost:11434",
        num_ctx: int = 8192,
        model_name: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name or model_id
        self._transport = OllamaGenerateTransport(
            model_id=model_id,
            base_url=base_url,
            num_ctx=num_ctx,
            model_name=self.model_name,
            parameters=parameters,
            extractor=self.extract_code,
            result_factory=self._make_result,
        )

    async def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 768,
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
