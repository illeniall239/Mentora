# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import math


def softmax(xs: list[list[float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for row in xs:
        m = max(row)
        if m == -math.inf:
            raise ValueError("every entry of a row is -inf")
        exps = [math.exp(x - m) for x in row]
        total = sum(exps)
        out.append([e / total for e in exps])
    return out


def attention(
    Q: list[list[float]], K: list[list[float]], V: list[list[float]], causal: bool = False
) -> tuple[list[list[float]], list[list[float]]]:
    n, m = len(Q), len(K)
    if len(V) != m:
        raise ValueError("K and V must have the same number of rows")
    d_k = len(Q[0])
    if any(len(k) != d_k for k in K):
        raise ValueError("Q and K rows must have the same length")
    if causal and n != m:
        raise ValueError("causal attention needs as many queries as keys")
    scale = math.sqrt(d_k)
    scores = [[sum(q * k for q, k in zip(Q[i], K[j])) / scale for j in range(m)] for i in range(n)]
    if causal:
        for i in range(n):
            for j in range(i + 1, m):
                scores[i][j] = -math.inf
    weights = softmax(scores)
    d_v = len(V[0])
    output = [[sum(weights[i][j] * V[j][c] for j in range(m)) for c in range(d_v)] for i in range(n)]
    return output, weights
