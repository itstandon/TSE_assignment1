"""Violation rate and its 95% Wilson score interval (better behaved than the normal approximation near 0 and 1,
which is where most MRs will sit)."""
from __future__ import annotations

from math import sqrt


def wilson(violations: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. (nan, nan) when there are no trials."""
    if trials <= 0:
        return float("nan"), float("nan")
    p = violations / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)
