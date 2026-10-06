# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import math


def train_bigram(ids: list[int], vocab: int) -> list[list[float]]:
    if any(not 0 <= i < vocab for i in ids):
        raise ValueError("token id outside the vocabulary")
    counts = [[0] * vocab for _ in range(vocab)]
    for a, b in zip(ids, ids[1:]):
        counts[a][b] += 1
    table = []
    for row in counts:
        total = sum(row) + vocab
        table.append([(c + 1) / total for c in row])
    return table


def perplexity(model: list[list[float]], ids: list[int]) -> float:
    if len(ids) < 2:
        raise ValueError("need at least two tokens to score a bigram")
    nll = sum(-math.log(model[a][b]) for a, b in zip(ids, ids[1:]))
    return math.exp(nll / (len(ids) - 1))
