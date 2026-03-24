"""Expert score aggregation and Cohen's kappa."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Iterable, Mapping


RatingRow = Mapping[str, object]


def aggregate_expert_scores(
    rows: Iterable[RatingRow],
    criteria: list[str],
) -> dict[str, float]:
    """Aggregate mean scores across criteria."""
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for criterion in criteria:
            if criterion in row and row[criterion] is not None:
                values[criterion].append(float(row[criterion]))
    return {criterion: mean(scores) for criterion, scores in values.items() if scores}


def cohens_kappa(
    rater_a: list[int],
    rater_b: list[int],
    *,
    scale_min: int,
    scale_max: int,
) -> float:
    """Compute Cohen's kappa for two aligned rating lists."""
    if len(rater_a) != len(rater_b):
        raise ValueError("Rating lists must have equal length.")
    if not rater_a:
        return 0.0

    categories = list(range(scale_min, scale_max + 1))
    total = len(rater_a)
    observed = sum(a == b for a, b in zip(rater_a, rater_b, strict=True)) / total
    counts_a = Counter(rater_a)
    counts_b = Counter(rater_b)
    expected = sum((counts_a[c] / total) * (counts_b[c] / total) for c in categories)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)
