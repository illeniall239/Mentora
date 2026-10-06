# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import math


def make_xy(ids: list[int], block_size: int, i: int) -> tuple[list[int], list[int]]:
    if i < 0 or block_size < 1 or i + block_size + 1 > len(ids):
        raise ValueError("window does not fit in ids")
    return ids[i : i + block_size], ids[i + 1 : i + block_size + 1]


def cross_entropy(logits: list[list[float]], targets: list[int]) -> float:
    if not logits or len(logits) != len(targets):
        raise ValueError("logits and targets must be non-empty and the same length")
    total = 0.0
    for row, target in zip(logits, targets):
        m = max(row)
        lse = m + math.log(sum(math.exp(z - m) for z in row))
        total += lse - row[target]
    return total / len(logits)
