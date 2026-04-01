"""Pass@K helpers."""

from __future__ import annotations

from math import comb


def pass_at_k(results_per_query: list[list[bool]], k: int) -> float:
    """Compute unbiased Pass@K.

    If a query has fewer than `k` candidates, the metric uses the available
    candidate count for that query.
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
    """Compute Pass@K for every K in *k_values*.

    Args:
        results_per_query: For each query, a list of booleans (True = correct).
        k_values: List of K values to evaluate (e.g. [1, 5, 10]).

    Returns:
        Mapping ``{k: pass_at_k_score}``.

    Raises:
        ValueError: If ``k_values`` is empty, contains non-positive values, or
            contains duplicates.
    """
    if not k_values:
        raise ValueError("k_values must not be empty")
    if any(k <= 0 for k in k_values):
        raise ValueError(f"All k_values must be positive; got {k_values}")
    if len(set(k_values)) != len(k_values):
        raise ValueError(f"k_values must not contain duplicates; got {k_values}")
    return {k: pass_at_k(results_per_query, k) for k in k_values}
