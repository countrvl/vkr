"""Expert score aggregation and Cohen's kappa."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Mapping


RatingRow = Mapping[str, object]


_RATING_MIN = 1
_RATING_MAX = 5


def expert_score(completeness: float, efficiency: float, readability: float) -> float:
    """Compute Expert Score as the average of three criteria.

    Args:
        completeness: Completeness rating (1–5).
        efficiency: Efficiency rating (1–5).
        readability: Readability rating (1–5).

    Returns:
        ES = (C + E + R) / 3.

    Raises:
        ValueError: If any rating is outside [1, 5].
    """
    for name, value in (
        ("completeness", completeness),
        ("efficiency", efficiency),
        ("readability", readability),
    ):
        if not (_RATING_MIN <= value <= _RATING_MAX):
            raise ValueError(f"{name} must be in [{_RATING_MIN}, {_RATING_MAX}]; got {value}")
    return (completeness + efficiency + readability) / 3


@dataclass
class ExpertEvaluation:
    """Single expert evaluation of one SQL candidate."""

    sample_id: str
    completeness: int  # 1–5
    efficiency: int    # 1–5
    readability: int   # 1–5

    def __post_init__(self) -> None:
        for field_name, value in (
            ("completeness", self.completeness),
            ("efficiency", self.efficiency),
            ("readability", self.readability),
        ):
            if not (_RATING_MIN <= value <= _RATING_MAX):
                raise ValueError(
                    f"{field_name} must be in [{_RATING_MIN}, {_RATING_MAX}]; got {value}"
                )

    @property
    def score(self) -> float:
        """ES = (C + E + R) / 3. Delegates to :func:`expert_score`."""
        return expert_score(self.completeness, self.efficiency, self.readability)


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
    scale_min: int = 1,
    scale_max: int = 5,
) -> float:
    """Compute Cohen's kappa for two aligned rating lists.

    Uses ``sklearn.metrics.cohen_kappa_score`` when available; falls back to a
    manual implementation otherwise.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("Rating lists must have equal length.")
    if not rater_a:
        return 0.0

    try:
        from sklearn.metrics import cohen_kappa_score  # type: ignore[import]

        return float(cohen_kappa_score(rater_a, rater_b))
    except ImportError:
        pass

    categories = list(range(scale_min, scale_max + 1))
    total = len(rater_a)
    observed = sum(a == b for a, b in zip(rater_a, rater_b, strict=True)) / total
    counts_a = Counter(rater_a)
    counts_b = Counter(rater_b)
    expected = sum((counts_a[c] / total) * (counts_b[c] / total) for c in categories)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)
