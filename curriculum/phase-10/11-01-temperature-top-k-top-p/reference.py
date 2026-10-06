# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import math


def _renormalize(probs: list[float]) -> list[float]:
    total = sum(probs)
    return [x / total for x in probs]


def _descending_indices(probs: list[float]) -> list[int]:
    # sorted is stable, so equal probabilities keep their lower index first.
    return sorted(range(len(probs)), key=lambda i: -probs[i])


def apply_temperature(logits: list[float], T: float) -> list[float]:
    if T <= 0:
        raise ValueError("T must be positive")
    scaled = [x / T for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    return _renormalize(exps)


def top_k_filter(probs: list[float], k: int) -> list[float]:
    if k < 1:
        raise ValueError("k must be at least 1")
    keep = set(_descending_indices(probs)[:k])
    return _renormalize([x if i in keep else 0.0 for i, x in enumerate(probs)])


def top_p_filter(probs: list[float], p: float) -> list[float]:
    if p <= 0 or p > 1:
        raise ValueError("p must be in (0, 1]")
    keep: set[int] = set()
    cumulative = 0.0
    for i in _descending_indices(probs):
        keep.add(i)
        cumulative += probs[i]
        if cumulative >= p:
            break
    return _renormalize([x if i in keep else 0.0 for i, x in enumerate(probs)])
