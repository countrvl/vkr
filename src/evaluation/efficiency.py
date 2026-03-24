"""Efficiency metric aggregation."""

from __future__ import annotations

from statistics import mean
from typing import Any

from src.inference.base import GenerationResult


def compute_efficiency(results: list[GenerationResult], config: dict[str, Any]) -> dict[str, float | None]:
    """Compute Tinf, Mem, Tok, Cost, and an aggregate efficiency score.

    Local backends default to zero direct monetary cost. Missing pricing or
    missing memory measurements produce `None` for the affected component.
    """
    if not results:
        return {"Tinf": None, "Mem": None, "Tok": None, "Cost": None, "Eff": None}

    weights = config["efficiency_weights"]
    latency = mean(result.latency_ms for result in results)
    tokens = mean(result.tokens_input + result.tokens_output for result in results)
    memory_values = [
        float(result.metadata["memory_mb"])
        for result in results
        if result.metadata.get("memory_mb") is not None
    ]
    memory = mean(memory_values) if memory_values else None

    cost_values = []
    for result in results:
        backend = result.metadata.get("backend")
        if backend == "ollama":
            cost_values.append(0.0)
            continue
        if "cost_usd" in result.metadata:
            cost_values.append(float(result.metadata["cost_usd"]))
            continue
        pricing = result.metadata.get("pricing")
        if pricing:
            input_cost = (result.tokens_input / 1_000_000.0) * float(pricing.get("input_per_mtok", 0.0))
            output_cost = (result.tokens_output / 1_000_000.0) * float(pricing.get("output_per_mtok", 0.0))
            cost_values.append(input_cost + output_cost)
    cost = mean(cost_values) if len(cost_values) == len(results) else None

    components = {"Tinf": latency, "Mem": memory, "Tok": tokens, "Cost": cost}
    if any(value is None for value in components.values()):
        eff = None
    else:
        eff = (
            weights["alpha"] * components["Tinf"]
            + weights["beta"] * components["Mem"]
            + weights["gamma"] * components["Tok"]
            + weights["delta"] * components["Cost"]
        )
    return {**components, "Eff": eff}
