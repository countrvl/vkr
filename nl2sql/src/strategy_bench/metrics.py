"""Metrics for strategy-bench sample logs."""

from __future__ import annotations

from statistics import mean
from typing import Any

from nl2sql.src.evaluation.pass_at_k import pass_at_k

from .logger import CaseRunLog


def aggregate_case_logs(logs: list[CaseRunLog]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[CaseRunLog]] = {}
    for log in logs:
        grouped.setdefault(log.strategy, []).append(log)

    summary: dict[str, dict[str, Any]] = {}
    for strategy, strategy_logs in grouped.items():
        execution_successes = [bool(log.metrics.get("execution_success")) for log in strategy_logs]
        accuracy_values = [log.metrics.get("execution_accuracy") for log in strategy_logs]
        accuracy_denominator = [value for value in accuracy_values if value is not None]
        end_to_end_latencies = [float(log.metrics.get("end_to_end_latency_ms", 0.0)) for log in strategy_logs]
        model_latencies = [float(log.metrics.get("model_latency_ms", 0.0)) for log in strategy_logs]
        model_calls = [int(log.model_call_count) for log in strategy_logs]
        pass_inputs = [_pass_candidates(log) for log in strategy_logs]
        recovered = 0
        first_failures = 0
        for log in strategy_logs:
            candidates = _pass_candidates(log)
            if len(candidates) > 1 and not candidates[0]:
                first_failures += 1
                if any(candidates[1:]):
                    recovered += 1
        summary[strategy] = {
            "strategy": strategy,
            "n_cases": len(strategy_logs),
            "execution_success_rate": sum(execution_successes) / len(strategy_logs) if strategy_logs else 0.0,
            "execution_accuracy": (
                sum(bool(value) for value in accuracy_denominator) / len(accuracy_denominator)
                if accuracy_denominator
                else None
            ),
            "pass_at_3": pass_at_k(pass_inputs, 3) if pass_inputs else 0.0,
            "latency_ms": mean(end_to_end_latencies) if end_to_end_latencies else 0.0,
            "model_latency_ms": mean(model_latencies) if model_latencies else 0.0,
            "cost_model_calls": mean(model_calls) if model_calls else 0.0,
            "recovery_rate": (recovered / first_failures) if first_failures else 0.0,
        }
    return summary


def _pass_candidates(log: CaseRunLog) -> list[bool]:
    candidates = log.metrics.get("candidate_accuracies")
    if isinstance(candidates, list):
        return [bool(value) for value in candidates[:3]]
    accuracy = log.metrics.get("execution_accuracy")
    if accuracy is None:
        return [False]
    return [bool(accuracy)]
