"""Token usage extraction utilities."""

from __future__ import annotations

from typing import Any


def extract_usage(metadata: dict[str, Any]) -> dict[str, int | None]:
    """Extract normalized token counts from model metadata.

    Supports OpenAI-style `usage` and Ollama counters when present.
    """
    usage = metadata.get("usage")
    if isinstance(usage, dict):
        return {
            "prompt_tokens": _to_int_or_none(usage.get("prompt_tokens")),
            "completion_tokens": _to_int_or_none(usage.get("completion_tokens")),
            "total_tokens": _to_int_or_none(usage.get("total_tokens")),
        }

    prompt_eval = metadata.get("prompt_eval_count")
    eval_count = metadata.get("eval_count")
    prompt_tokens = _to_int_or_none(prompt_eval)
    completion_tokens = _to_int_or_none(eval_count)
    total_tokens = None
    if prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def aggregate_usage(usages: list[dict[str, int | None]]) -> dict[str, int]:
    """Aggregate token usage values, treating None as 0."""
    total_prompt = sum((u.get("prompt_tokens") or 0) for u in usages)
    total_completion = sum((u.get("completion_tokens") or 0) for u in usages)
    total_all = sum((u.get("total_tokens") or 0) for u in usages)
    return {
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "total_tokens": total_all,
    }


def _to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
