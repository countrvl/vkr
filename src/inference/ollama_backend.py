"""Async Ollama backend for local NL2SQL models."""

from __future__ import annotations

import asyncio
import subprocess
import time
from typing import Any

import httpx

from src.inference.base import GenerationResult, InferenceBackend, extract_sql


class OllamaInferenceBackend(InferenceBackend):
    """Inference backend for Ollama's `/api/generate` endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        model_name: str,
        parameters: dict[str, Any] | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._model_name = model_name
        self._parameters = parameters or {}
        self._timeout = timeout

    async def generate(
        self,
        prompt: str,
        n: int = 1,
        temperature: float = 0.0,
    ) -> list[GenerationResult]:
        results: list[GenerationResult] = []
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for _ in range(n):
                started_at = time.perf_counter()
                response = await client.post(
                    f"{self._base_url}/api/generate",
                    json={
                        "model": self._model_id,
                        "prompt": prompt,
                        "stream": False,
                        "options": {**self._parameters, "temperature": temperature},
                    },
                )
                response.raise_for_status()
                payload = response.json()
                latency_ms = (time.perf_counter() - started_at) * 1000.0
                raw_response = str(payload.get("response", ""))
                results.append(
                    GenerationResult(
                        sql=extract_sql(raw_response),
                        raw_response=raw_response,
                        tokens_input=int(payload.get("prompt_eval_count", 0) or 0),
                        tokens_output=int(payload.get("eval_count", 0) or 0),
                        latency_ms=latency_ms,
                        model_name=self._model_name,
                        metadata={
                            "backend": "ollama",
                            "memory_mb": await _probe_gpu_memory_mb(),
                        },
                    )
                )
        return results


async def _probe_gpu_memory_mb() -> float | None:
    """Return current used GPU memory in MB if `nvidia-smi` is available."""
    try:
        process = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await process.communicate()
        if process.returncode != 0:
            return None
        line = stdout.decode("utf-8").strip().splitlines()[0]
        return float(line)
    except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
        return None
