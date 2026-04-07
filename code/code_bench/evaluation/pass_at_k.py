"""Prefix-based Pass@K helpers for code generation."""

from __future__ import annotations

def pass_at_k(results_per_query: list[list[bool]], k: int) -> float:
    """Compute prefix-based Pass@K.

    A query counts as solved if any of the first ``k`` candidates is correct.
    This matches the experiment definition used in the code-generation domain.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not results_per_query:
        return 0.0

    scores = []
    for query_results in results_per_query:
        if not query_results:
            scores.append(0.0)
            continue
        prefix = query_results[:k]
        scores.append(1.0 if any(bool(value) for value in prefix) else 0.0)
    return sum(scores) / len(scores)


def compute_all_pass_at_k(results_per_query: list[list[bool]], k_values: list[int]) -> dict[int, float]:
    """Compute Pass@K for every K in *k_values*."""
    if not k_values:
        raise ValueError("k_values must not be empty")
    if any(k <= 0 for k in k_values):
        raise ValueError(f"All k_values must be positive; got {k_values}")
    if len(set(k_values)) != len(k_values):
        raise ValueError(f"k_values must not contain duplicates; got {k_values}")
    return {k: pass_at_k(results_per_query, k) for k in k_values}
