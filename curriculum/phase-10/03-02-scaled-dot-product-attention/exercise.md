# Scaled dot-product attention

Topic: 3. Attention and self-attention from scratch
Difficulty: 2 of 3

## Problem

Write attention as it appears in the Transformer paper, on plain lists of lists, in pure Python (no numpy, no torch). Each query row looks up a soft mixture of value rows, weighted by how well the query matches each key.

`attention(Q: list[list[float]], K: list[list[float]], V: list[list[float]], causal: bool = False) -> tuple[list[list[float]], list[list[float]]]`

- `Q` has `n` rows of `d_k` numbers, `K` has `m` rows of `d_k`, `V` has `m` rows of `d_v`.
- Scores: `S[i][j] = (Q[i] · K[j]) / sqrt(d_k)`.
- If `causal`, set `S[i][j] = -inf` for every `j > i` **before** the softmax. Causal attention requires `n == m`.
- Weights: `W = softmax(S)` row by row (subtract the row max; `-inf` becomes exactly 0). You may copy your `softmax` from `03-01` into this file; do not import from other exercises.
- Output: `O[i] = Σ_j W[i][j] · V[j]`, so `O` has `n` rows of `d_v`.
- Return `(O, W)`. Every row of `W` sums to 1 within `1e-9`. With `causal=True`, `W[0]` is `[1, 0, 0, …]` and `O[0]` equals `V[0]` exactly.
- Raise `ValueError` if `K` and `V` have different numbers of rows, if the rows of `Q` and `K` have different lengths, or if `causal` is requested with `n != m`.

Tests check against `torch.nn.functional.scaled_dot_product_attention` on the same inputs within `1e-6`.

## Examples

```
Q = K = [[1, 0], [0, 1], [1, 1]]      V = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]      d_k = 2

attention(Q, K, V, causal=True) →
  O = [[1.0, 0.0, 0.0],
       [0.3302, 0.6698, 0.0],
       [0.2482, 0.2482, 0.5035]]
  W = [[1.0, 0.0, 0.0],
       [0.3302, 0.6698, 0.0],
       [0.2482, 0.2482, 0.5035]]          W == O here because V is the identity

attention(Q, K, V, causal=False)[1] → [[0.1978, 0.4011, 0.4011], ...]     row 1 now sees key 2
attention([[1, 0]], [[1, 0], [0, 1]], [[1], [2], [3]])   → ValueError     K has 2 rows, V has 3
```

## Constraints

- `n, m ≤ 16`, `d_k, d_v ≤ 16`.
- Match the PyTorch reference within `1e-6`.
- Pure Python only; `math` is allowed.

## Hints

1. What are the shapes of `S`, `W` and `O` in terms of `n`, `m`, `d_k` and `d_v`? Which axis does each softmax run over: the queries or the keys?
2. Why divide by `sqrt(d_k)` and not by `d_k`? What is the variance of a dot product of two vectors with `d_k` independent unit-variance entries?
3. Which entries of `S` must be `-inf` for query `i` so that it never sees the future, and what would happen to the row sum if you set them to 0 instead?
4. `O[i]` is a weighted sum of *which* rows: rows of `V`, or rows of `K`? What do `K` and `V` each contribute to the lookup?

## Explain-back

- In self-attention, are `Q`, `K` and `V` three different inputs? Where do they come from, and how would cross-attention change that?
- Why must the causal mask be applied before the softmax rather than by zeroing weights after it? Show with one row of three scores what the two give.
- Attention weights are often plotted as "what the model looked at". Why is a high weight not an explanation of the model's reasoning?
- Did attention begin with the Transformer? What did the earlier RNN encoder-decoder attention (Bahdanau, 2014) attend over, and what did the 2017 paper remove?
