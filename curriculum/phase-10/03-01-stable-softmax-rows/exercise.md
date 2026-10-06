# Stable row-wise softmax

Topic: 3. Attention and self-attention from scratch
Difficulty: 1 of 3

## Problem

Attention turns every row of scores into a probability distribution with a softmax. Done naively, `exp` overflows for large scores and underflows to `0/0` for very negative ones. Write it the stable way, in pure Python (no numpy, no torch).

`softmax(xs: list[list[float]]) -> list[list[float]]` returns a new list of rows where row `i` is `exp(x_ij − m_i) / Σ_j exp(x_ij − m_i)` with `m_i = max(row i)`. Subtracting the row's maximum changes nothing mathematically but keeps every `exp` argument at most 0.

- Rows are independent: a row's result never depends on another row.
- A score of `-math.inf` (a masked position) gets **exactly** `0.0` and still leaves the rest of the row summing to 1.
- Every output row sums to 1 within `1e-9`.
- Raise `ValueError` for an empty row or a row where every entry is `-inf` (no probability mass can be assigned).
- Do not modify `xs`.

## Examples

```
softmax([[1000.0, 1001.0]])            → [[0.2689, 0.7311]]     naive exp(1000) overflows
softmax([[-1000.0, -1001.0]])          → [[0.7311, 0.2689]]     naive exp underflows to 0/0
softmax([[1.0, 2.0, -inf]])            → [[0.2689, 0.7311, 0.0]]
softmax([[0.0, 0.0, 0.0, 0.0]])        → [[0.25, 0.25, 0.25, 0.25]]
softmax([[5.0], [3.0, 3.0]])           → [[1.0], [0.5, 0.5]]
softmax([[-inf, -inf]])                → ValueError
```

## Constraints

- At most 64 rows of at most 64 entries.
- Results are compared within `1e-9`.
- Pure Python; `math.exp` and `math.inf` are allowed.

## Hints

1. What is the largest value `math.exp` can return before it raises `OverflowError`? What is the largest argument you get after subtracting the row's max?
2. Multiply numerator and denominator of `exp(x_j) / Σ exp(x_k)` by the same constant `exp(−m)`. What changes?
3. What is `math.exp(-math.inf)`? What does that give you for free at a masked position?
4. `max` of a row whose entries are all `-inf` is `-inf`. What would `x_j − m` be there, and how should you detect that case before it happens?

## Explain-back

- Why does softmax need the max trick while a plain normalization `x / sum(x)` does not? What else does `x / sum(x)` get wrong for attention scores?
- Attention scores are divided by `√d_k` before the softmax. What happens to a softmax over scores in the hundreds, and why is that bad for learning?
- A masked position gets `-inf` *before* the softmax. What would go wrong if you instead set its probability to 0 *after* the softmax?
- If one row of scores has a single `-inf` and another has none, is the result of one row affected by the other? What does that say about the batch dimension in attention?
