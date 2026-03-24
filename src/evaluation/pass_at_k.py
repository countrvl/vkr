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
