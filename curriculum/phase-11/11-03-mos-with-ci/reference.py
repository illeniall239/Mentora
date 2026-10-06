# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import math

Z95 = 1.96


def mos(ratings: list[float]) -> tuple[float, float, float]:
    n = len(ratings)
    if n < 2:
        raise ValueError("a confidence interval needs at least 2 ratings")
    if any(not 1.0 <= r <= 5.0 for r in ratings):
        raise ValueError("ratings must lie on the 1..5 MOS scale")
    mean = sum(ratings) / n
    variance = sum((r - mean) ** 2 for r in ratings) / (n - 1)
    margin = Z95 * math.sqrt(variance) / math.sqrt(n)
    return mean, mean - margin, mean + margin
