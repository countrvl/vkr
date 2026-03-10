"""Latency helpers for model inference."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


def timed_call(func: Callable[..., tuple[str, dict[str, Any]]], *args: Any, **kwargs: Any) -> tuple[str, dict[str, Any], float]:
    """Measure call latency in seconds."""
    start = time.perf_counter()
    text, metadata = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return text, metadata, elapsed


def average_latency(latencies: list[float]) -> float:
    """Compute average latency in seconds."""
    if not latencies:
        return 0.0
    return sum(latencies) / len(latencies)
