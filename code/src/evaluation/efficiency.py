"""Efficiency metric aggregation for code generation."""

from __future__ import annotations

import copy
from statistics import mean
from typing import Any

from code.src.inference.base import GenerationResult


_EFF_COMPONENTS = ("Tinf", "Mem", "Tok", "Cost")


def _validate_weights(weights: dict[str, Any]) -> None:
    total = sum(weights[k] for k in ("alpha", "beta", "gamma", "delta"))
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"efficiency_weights must sum to 1.0 for a valid convex combination; got {total:.6f}"
        )


def compute_efficiency(results: list[GenerationResult], config: dict[str, Any]) -> dict[str, float | None]:
    """Compute Tinf, Mem, Tok, Cost, and an aggregate efficiency score."""
    if not results:
        return {"Tinf": None, "Mem": None, "Tok": None, "Cost": None, "Eff": None}

    weights = config["efficiency_weights"]
    _validate_weights(weights)

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
        cost_usd = result.metadata.get("cost_usd")
        if cost_usd is not None:
            cost_values.append(float(cost_usd))
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


def normalize_efficiency_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a new list of rows with min-max normalized Eff components."""
    result = [copy.copy(row) for row in rows]
    for component in _EFF_COMPONENTS:
        values = [float(row[component]) for row in result if row.get(component) is not None]
        if not values:
            for row in result:
                row[f"{component}_norm"] = None
            continue
        lo, hi = min(values), max(values)
        span = hi - lo
        for row in result:
            raw = row.get(component)
            if raw is None:
                row[f"{component}_norm"] = None
            else:
                row[f"{component}_norm"] = 0.0 if span == 0 else (float(raw) - lo) / span

    for row in result:
        weights = row.get("_weights")
        if not weights:
            continue
        _validate_weights(weights)
        norms = [row.get(f"{c}_norm") for c in _EFF_COMPONENTS]
        if any(v is None for v in norms):
            row["Eff_normalized"] = None
        else:
            tinf_n, mem_n, tok_n, cost_n = norms
            row["Eff_normalized"] = (
                weights["alpha"] * tinf_n
                + weights["beta"] * mem_n
                + weights["gamma"] * tok_n
                + weights["delta"] * cost_n
            )
    return result
