# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import math


def pass_at_k(n: int, c: int, k: int) -> float:
    if n < 1 or not 1 <= k <= n or not 0 <= c <= n:
        raise ValueError("need n >= 1, 1 <= k <= n and 0 <= c <= n")
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_at_k_from_results(results: list[bool], k: int) -> float:
    if not results:
        raise ValueError("need at least one sampled result")
    return pass_at_k(len(results), sum(1 for r in results if r), k)
