# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import math


def frechet_distance_1d(mu1: float, s1: float, mu2: float, s2: float) -> float:
    if s1 < 0 or s2 < 0:
        raise ValueError("variances must be non-negative")
    return (mu1 - mu2) ** 2 + s1 + s2 - 2.0 * math.sqrt(s1 * s2)
