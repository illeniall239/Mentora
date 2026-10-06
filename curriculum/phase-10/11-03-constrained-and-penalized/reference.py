# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import math


def softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]


def constrained_mask(logits: list[float], allowed_ids: set[int]) -> list[float]:
    if not allowed_ids:
        raise ValueError("allowed_ids must not be empty")
    if any(i < 0 or i >= len(logits) for i in allowed_ids):
        raise ValueError("allowed id outside the vocabulary")
    return [x if i in allowed_ids else -math.inf for i, x in enumerate(logits)]


def repetition_penalty(logits: list[float], history: list[int], penalty: float) -> list[float]:
    if penalty <= 0:
        raise ValueError("penalty must be positive")
    out = list(logits)
    for i in set(history):
        if 0 <= i < len(out):
            out[i] = out[i] / penalty if out[i] > 0 else out[i] * penalty
    return out
