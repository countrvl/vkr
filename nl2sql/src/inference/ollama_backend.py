"""Async Ollama backend for local NL2SQL models."""

from __future__ import annotations

import httpx  # noqa: F401 - kept for tests that monkeypatch the shared transport client
from typing import Any

from nl2sql.src.inference.base import GenerationResult, InferenceBackend, SQL_RESPONSE_SCHEMA
from shared.inference.ollama_transport import OllamaGenerateTransport


class OllamaBackend(InferenceBackend):
    """Inference backend for Ollama's `/api/generate` endpoint."""

    def __init__(
        self,
        model_id: str,
        base_url: str = "http://localhost:11434",
        num_ctx: int = 4096,
        model_name: str | None = None,
        parameters: dict[str, Any] | None = None,
        structured_output: bool = True,
    ) -> None:
        """Initialize the Ollama backend.

        Args:
            model_id: Ollama model tag.
            base_url: Ollama REST API base URL.
            num_ctx: Context window size.
            model_name: Human-readable model name.
            parameters: Additional Ollama options.
        """
        self.model_name = model_name or model_id
        self._transport = OllamaGenerateTransport(
            model_id=model_id,
            base_url=base_url,
            num_ctx=num_ctx,
            model_name=self.model_name,
            parameters=parameters,
            extractor=self.extract_sql,
            result_factory=self._make_result,
            format_schema=SQL_RESPONSE_SCHEMA if structured_output else None,
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
        """Generate SQL candidates sequentially through Ollama."""
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
            sql=extracted,
            raw_response=raw_response,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=latency_ms,
            model_name=model_name,
            metadata=metadata,
        )


OllamaInferenceBackend = OllamaBackend
