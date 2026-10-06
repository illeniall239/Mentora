# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import random


def sample(probs: list[float], rng: random.Random) -> int:
    if not probs or any(p < 0 for p in probs) or abs(sum(probs) - 1.0) > 1e-6:
        raise ValueError("probs must be a non-empty distribution summing to 1")
    u = rng.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if cumulative > u:
            return i
    return max(i for i, p in enumerate(probs) if p > 0)
