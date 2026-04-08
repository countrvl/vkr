"""Вспомогательные функции `Pass@K`."""

from __future__ import annotations

from math import comb


def pass_at_k(results_per_query: list[list[bool]], k: int) -> float:
    """Посчитать unbiased `Pass@K`.

    Если у задачи меньше `k` кандидатов, используется фактическое число кандидатов.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not results_per_query:
        return 0.0

    scores = []
    for query_results in results_per_query:
        n = len(query_results)
        if n == 0:
            scores.append(0.0)
            continue
        effective_k = min(k, n)
        c = sum(bool(value) for value in query_results)
        if n - c < effective_k:
            scores.append(1.0)
        else:
            scores.append(1.0 - (comb(n - c, effective_k) / comb(n, effective_k)))
    return sum(scores) / len(scores)


def compute_all_pass_at_k(results_per_query: list[list[bool]], k_values: list[int]) -> dict[int, float]:
    """Посчитать `Pass@K` для каждого значения из `k_values`."""
    if not k_values:
        raise ValueError("k_values must not be empty")
    if any(k <= 0 for k in k_values):
        raise ValueError(f"All k_values must be positive; got {k_values}")
    if len(set(k_values)) != len(k_values):
        raise ValueError(f"k_values must not contain duplicates; got {k_values}")
    return {k: pass_at_k(results_per_query, k) for k in k_values}
