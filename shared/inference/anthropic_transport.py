"""Общий Anthropic transport для online- и batch-инференса."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

import httpx


LOGGER = logging.getLogger(__name__)
ResultT = TypeVar("ResultT")
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
POLL_DELAYS_SECONDS = (2.0, 3.0, 5.0)
DEFAULT_MAX_BATCH_WAIT_SECONDS = 1800.0


class AnthropicMessagesTransport:
    """Переиспользуемый transport для Anthropic Messages API."""

    supports_batch = True

    def __init__(
        self,
        *,
        model_id: str,
        base_url: str,
        api_key: str,
        model_name: str,
        parameters: dict[str, Any] | None = None,
        pricing: dict[str, Any] | None = None,
        extractor: Callable[[str], str],
        result_factory: Callable[..., ResultT],
        use_batch: bool = True,
    ) -> None:
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.parameters = dict(parameters or {})
        self.pricing = self._normalize_pricing(pricing)
        self.extractor = extractor
        self.result_factory = result_factory
        self.use_batch = use_batch
        self.anthropic_version = str(
            self.parameters.pop("anthropic_version", DEFAULT_ANTHROPIC_VERSION)
        )
        self.batch_pricing_multiplier = float(self.parameters.pop("batch_pricing_multiplier", 0.5))
        self.max_batch_wait_seconds = float(
            self.parameters.pop("max_batch_wait_seconds", DEFAULT_MAX_BATCH_WAIT_SECONDS)
        )
        self.status_callback: Callable[[dict[str, Any]], None] | None = None
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": self.anthropic_version,
                "content-type": "application/json",
            },
            timeout=httpx.Timeout(300.0),
        )

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

    async def aclose(self) -> None:
        await self.client.aclose()

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
        if n != 1:
            raise NotImplementedError("Anthropic transport in v1 supports only single-shot generation.")
        return [await self._generate_one(prompt, temperature, max_tokens, seed, top_p)]

    async def generate_batch(
        self,
        *,
        prompts: list[str],
        temperature: float,
        max_tokens: int,
        seed: int | None,
        top_p: float | None,
    ) -> list[list[ResultT] | Exception]:
        if not prompts:
            return []
        if not self.use_batch:
            return await self._generate_online_many(prompts, temperature, max_tokens, seed, top_p)
        try:
            (
                batch_id,
                batch_state,
                custom_ids,
                started_at,
            ) = await self._submit_batch_native(
                prompts=prompts,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                top_p=top_p,
            )
        except Exception as exc:
            LOGGER.warning(
                "Anthropic batch submit failed for %s: %s. Falling back to online requests.",
                self.model_id,
                exc,
            )
            return await self._generate_online_many(prompts, temperature, max_tokens, seed, top_p)
        return await self._await_batch_results(
            batch_id=batch_id,
            batch_state=batch_state,
            custom_ids=custom_ids,
            prompts=prompts,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            started_at=started_at,
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
    ) -> list[list[ResultT] | Exception]:
        del seed
        if not prompts:
            return []
        custom_ids = [f"req_{idx:06d}" for idx in range(len(prompts))]
        batch_state = await self._request_json("GET", f"/v1/messages/batches/{batch_id}")
        self._notify_status(
            {
                "phase": "resuming",
                "batch_id": batch_id,
                "status": batch_state.get("processing_status") or batch_state.get("status") or "unknown",
                "n_requests": len(prompts),
            }
        )
        return await self._await_batch_results(
            batch_id=batch_id,
            batch_state=batch_state,
            custom_ids=custom_ids,
            prompts=prompts,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            started_at=None,
        )

    async def _generate_online_many(
        self,
        prompts: list[str],
        temperature: float,
        max_tokens: int,
        seed: int | None,
        top_p: float | None,
    ) -> list[list[ResultT] | Exception]:
        results: list[list[ResultT] | Exception] = []
        for prompt in prompts:
            try:
                result = await self._generate_one(prompt, temperature, max_tokens, seed, top_p)
            except Exception as exc:
                results.append(exc)
            else:
                results.append([result])
        return results

    async def _generate_one(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        top_p: float | None,
    ) -> ResultT:
        del seed  # Anthropic Messages API не поддерживает seed в текущем контуре.
        payload = self._build_message_params(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        started_at = time.perf_counter()
        response = await self._request_json("POST", "/v1/messages", json_payload=payload)
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        return self._result_from_message(
            message=response,
            latency_ms=latency_ms,
            metadata={
                "backend": "anthropic",
                "provider": "anthropic",
                "dispatch": "online",
            },
            pricing_multiplier=1.0,
        )

    async def _submit_batch_native(
        self,
        prompts: list[str],
        temperature: float,
        max_tokens: int,
        seed: int | None,
        top_p: float | None,
    ) -> tuple[str, dict[str, Any], list[str], float]:
        del seed
        started_at = time.perf_counter()
        requests: list[dict[str, Any]] = []
        custom_ids: list[str] = []
        for idx, prompt in enumerate(prompts):
            custom_id = f"req_{idx:06d}"
            custom_ids.append(custom_id)
            requests.append(
                {
                    "custom_id": custom_id,
                    "params": self._build_message_params(
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                    ),
                }
            )

        batch_response = await self._request_json(
            "POST",
            "/v1/messages/batches",
            json_payload={"requests": requests},
        )
        batch_id = str(batch_response["id"])
        batch_state = batch_response
        self._notify_status(
            {
                "phase": "submitted",
                "batch_id": batch_id,
                "status": batch_state.get("processing_status") or batch_state.get("status") or "submitted",
                "n_requests": len(prompts),
            }
        )
        return batch_id, batch_state, custom_ids, started_at

    async def _await_batch_results(
        self,
        *,
        batch_id: str,
        batch_state: dict[str, Any],
        custom_ids: list[str],
        prompts: list[str],
        temperature: float,
        max_tokens: int,
        top_p: float | None,
        started_at: float | None,
    ) -> list[list[ResultT] | Exception]:
        results_url = batch_state.get("results_url")
        poll_started_at = time.perf_counter()
        while not results_url:
            status = str(batch_state.get("processing_status") or batch_state.get("status") or "")
            elapsed_seconds = time.perf_counter() - poll_started_at
            self._notify_status(
                {
                    "phase": "polling",
                    "batch_id": batch_id,
                    "status": status or "unknown",
                    "elapsed_seconds": round(elapsed_seconds, 1),
                    "n_requests": len(prompts),
                }
            )
            if status.lower() in {"ended", "completed"}:
                results_url = batch_state.get("results_url")
                break
            if status.lower() in {"errored", "canceled", "cancelled", "expired"}:
                raise RuntimeError(f"Anthropic batch {batch_id} finished with status={status!r}.")
            if elapsed_seconds > self.max_batch_wait_seconds:
                raise TimeoutError(
                    f"Anthropic batch {batch_id} exceeded max wait of "
                    f"{self.max_batch_wait_seconds:.0f}s with status={status!r}."
                )
            await asyncio.sleep(POLL_DELAYS_SECONDS[min(len(custom_ids), len(POLL_DELAYS_SECONDS)) - 1])
            batch_state = await self._request_json("GET", f"/v1/messages/batches/{batch_id}")
            results_url = batch_state.get("results_url")

        if not results_url:
            raise RuntimeError(f"Anthropic batch {batch_id} completed without results_url.")

        self._notify_status(
            {
                "phase": "downloading_results",
                "batch_id": batch_id,
                "status": batch_state.get("processing_status") or batch_state.get("status") or "ended",
                "results_url": results_url,
                "n_requests": len(prompts),
            }
        )

        lines = await self._request_jsonl(results_url)
        total_latency_ms = self._estimate_batch_latency_ms(batch_state, started_at)
        per_request_latency_ms = total_latency_ms / max(len(prompts), 1)
        by_custom_id: dict[str, list[ResultT] | Exception] = {}

        for payload in lines:
            custom_id = str(payload.get("custom_id") or "")
            result = payload.get("result") or {}
            result_type = str(result.get("type") or "")
            if result_type == "succeeded":
                message = result.get("message") or {}
                by_custom_id[custom_id] = [
                    self._result_from_message(
                        message=message,
                        latency_ms=per_request_latency_ms,
                        metadata={
                            "backend": "anthropic",
                            "provider": "anthropic",
                            "dispatch": "batch",
                            "batch_id": batch_id,
                            "custom_id": custom_id,
                        },
                        pricing_multiplier=self.batch_pricing_multiplier,
                    )
                ]
            else:
                error_obj = result.get("error") or payload.get("error") or result or payload
                error_message = json.dumps(error_obj, ensure_ascii=False)
                by_custom_id[custom_id] = RuntimeError(
                    f"Anthropic batch item {custom_id} failed: {error_message}"
                )

        ordered: list[list[ResultT] | Exception] = []
        for custom_id, prompt in zip(custom_ids, prompts):
            item = by_custom_id.get(custom_id)
            if item is None:
                try:
                    fallback_result = await self._generate_one(prompt, temperature, max_tokens, None, top_p)
                except Exception as exc:
                    ordered.append(exc)
                else:
                    ordered.append([fallback_result])
            else:
                ordered.append(item)
        self._notify_status(
            {
                "phase": "completed",
                "batch_id": batch_id,
                "status": "completed",
                "n_requests": len(prompts),
            }
        )
        return ordered

    @staticmethod
    def _estimate_batch_latency_ms(batch_state: dict[str, Any], started_at: float | None) -> float:
        created_at = batch_state.get("created_at")
        if isinstance(created_at, str):
            try:
                created_dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                if created_dt.tzinfo is None:
                    created_dt = created_dt.replace(tzinfo=timezone.utc)
                return max((time.time() - created_dt.timestamp()) * 1000.0, 0.0)
            except ValueError:
                pass
        if started_at is not None:
            return (time.perf_counter() - started_at) * 1000.0
        return 0.0

    def _build_message_params(
        self,
        *,
        prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float | None,
    ) -> dict[str, Any]:
        params = dict(self.parameters)
        params.pop("max_tokens", None)
        payload = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            **params,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        return payload

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = await self.client.request(method, path, json=json_payload)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"Expected JSON object from Anthropic, got {type(payload).__name__}.")
        return payload

    async def _request_jsonl(self, url: str) -> list[dict[str, Any]]:
        response = await self.client.get(url)
        response.raise_for_status()
        payloads: list[dict[str, Any]] = []
        for line in response.text.splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise RuntimeError("Expected JSON object in Anthropic batch results line.")
            payloads.append(payload)
        return payloads

    def _result_from_message(
        self,
        *,
        message: dict[str, Any],
        latency_ms: float,
        metadata: dict[str, Any],
        pricing_multiplier: float,
    ) -> ResultT:
        raw_response = self._extract_text_content(message.get("content"))
        extracted = self.extractor(raw_response)
        usage = message.get("usage") or {}
        prompt_tokens = int(usage.get("input_tokens") or 0)
        output_tokens = int(usage.get("output_tokens") or 0)
        cost_usd = None
        metadata_out = dict(metadata)
        if self.pricing:
            cost_usd = (
                (prompt_tokens / 1_000_000.0) * float(self.pricing.get("input_per_mtok", 0.0))
                + (output_tokens / 1_000_000.0) * float(self.pricing.get("output_per_mtok", 0.0))
            ) * pricing_multiplier
            metadata_out["pricing"] = dict(self.pricing)
            metadata_out["pricing_multiplier"] = pricing_multiplier
            metadata_out["cost_usd"] = cost_usd
        return self.result_factory(
            extracted=extracted,
            raw_response=raw_response,
            tokens_input=prompt_tokens,
            tokens_output=output_tokens,
            latency_ms=latency_ms,
            model_name=self.model_name,
            metadata=metadata_out,
        )

    @staticmethod
    def _extract_text_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts).strip()

    def _notify_status(self, payload: dict[str, Any]) -> None:
        if self.status_callback is None:
            return
        try:
            self.status_callback(payload)
        except Exception:
            LOGGER.debug("Failed to handle Anthropic batch status callback.", exc_info=True)
