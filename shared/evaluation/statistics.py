"""Непараметрические summary-статистики для экспериментальных метрик."""

from __future__ import annotations

import math
import random
from typing import Callable, Iterable, Sequence, TypeVar


T = TypeVar("T")


def quantile(values: Sequence[float], q: float) -> float | None:
    """Линейно интерполированный квантиль для `q in [0, 1]`."""
    if not values:
        return None
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1]; got {q}")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def quantile_fields(
    values: Sequence[float],
    *,
    prefix: str,
    quantiles: Sequence[float],
) -> dict[str, float | None]:
    """Вернуть поля вида `{prefix}_q05`, `{prefix}_q50`, ..."""
    result: dict[str, float | None] = {}
    for q in quantiles:
        label = f"{int(round(q * 100)):02d}"
        result[f"{prefix}_q{label}"] = quantile(values, q)
    return result


def wilson_interval(successes: int, total: int, confidence_level: float = 0.95) -> tuple[float | None, float | None]:
    """Wilson CI для биномиальной доли."""
    if total <= 0:
        return None, None
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1); got {confidence_level}")
    z = _z_value_for_confidence(confidence_level)
    p_hat = successes / total
    denom = 1.0 + (z * z) / total
    center = (p_hat + (z * z) / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / total) + ((z * z) / (4.0 * total * total)))
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_interval(
    samples: Sequence[T],
    statistic_fn: Callable[[Sequence[T]], float],
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float | None, float | None]:
    """Percentile bootstrap CI для произвольной sample-level статистики."""
    if not samples:
        return None, None
    if len(samples) == 1:
        value = statistic_fn(samples)
        return value, value
    rng = random.Random(seed)
    estimates: list[float] = []
    size = len(samples)
    for _ in range(n_resamples):
        resample = [samples[rng.randrange(size)] for _ in range(size)]
        estimates.append(float(statistic_fn(resample)))
    alpha = (1.0 - confidence_level) / 2.0
    return quantile(estimates, alpha), quantile(estimates, 1.0 - alpha)


def bootstrap_distribution(
    samples: Sequence[T],
    statistic_fn: Callable[[Sequence[T]], float],
    *,
    n_resamples: int = 1000,
    seed: int = 42,
) -> list[float]:
    """Вернуть bootstrap-распределение оценки статистики."""
    if not samples:
        return []
    if len(samples) == 1:
        return [float(statistic_fn(samples))]
    rng = random.Random(seed)
    estimates: list[float] = []
    size = len(samples)
    for _ in range(n_resamples):
        resample = [samples[rng.randrange(size)] for _ in range(size)]
        estimates.append(float(statistic_fn(resample)))
    return estimates


def bootstrap_quantile_fields(
    samples: Sequence[T],
    statistic_fn: Callable[[Sequence[T]], float],
    *,
    prefix: str,
    quantiles: Sequence[float],
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, float | None]:
    """Вернуть квантильные поля для bootstrap-распределения статистики."""
    estimates = bootstrap_distribution(
        samples,
        statistic_fn,
        n_resamples=n_resamples,
        seed=seed,
    )
    return quantile_fields(estimates, prefix=prefix, quantiles=quantiles)


def _z_value_for_confidence(confidence_level: float) -> float:
    alpha = 1.0 - confidence_level
    # Достаточно для 90/95/99%; при других значениях будет interpolation via erf^-1 unavailable.
    lookup = {
        0.10: 1.6448536269514722,
        0.05: 1.959963984540054,
        0.01: 2.5758293035489004,
    }
    for tail_alpha, z in lookup.items():
        if abs(alpha - tail_alpha) < 1e-12:
            return z
    raise ValueError(
        "Unsupported confidence level. Use one of 0.90, 0.95, 0.99 "
        f"(got {confidence_level})."
    )
