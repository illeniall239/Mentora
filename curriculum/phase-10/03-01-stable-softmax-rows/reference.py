# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import math


def softmax(xs: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for row in xs:
        if not row:
            raise ValueError("cannot softmax an empty row")
        m = max(row)
        if m == -math.inf:
            raise ValueError("every entry of a row is -inf")
        exps = [math.exp(x - m) for x in row]
        total = sum(exps)
        out.append([e / total for e in exps])
    return out
