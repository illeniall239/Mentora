# Permutation equivariance

Topic: 3. Attention and self-attention from scratch
Difficulty: 2 of 3

## Problem

Self-attention has no idea where a token sits: shuffle the input rows and the output rows shuffle identically. Build a small numpy harness that demonstrates this, and shows what breaks it.

Write three functions:

- `self_attention(X: np.ndarray, params: dict[str, np.ndarray], causal: bool = False) -> np.ndarray`. `X` is `(n, d)`. `params` holds `"W_q"` and `"W_k"` of shape `(d, d_k)` and `"W_v"` of shape `(d, d_v)`. Compute `Q = X @ W_q`, `K = X @ W_k`, `V = X @ W_v`, then `softmax(Q Kᵀ / √d_k) V` with a row-wise softmax that subtracts the row max. If `causal`, set the scores with `j > i` to `-inf` before the softmax. Return `(n, d_v)`. Tests compare against `torch.nn.functional.scaled_dot_product_attention` within `1e-6`.
- `permute_rows(X: np.ndarray, perm: list[int]) -> np.ndarray` returns a **new** array whose row `i` is `X[perm[i]]`. Raise `ValueError` unless `perm` is a permutation of `0 … n-1`.
- `is_permutation_equivariant(fn, X: np.ndarray, perm: list[int], atol: float = 1e-6) -> bool` returns whether `fn(permute_rows(X, perm))` equals `permute_rows(fn(X), perm)` within `atol` (use `np.allclose`). `fn` maps an `(n, d)` array to an `(n, d')` array. Note that this is *equivariance* (the output moves with the input), not *invariance* (the output is unchanged).

The harness must then show, in the tests: non-causal self-attention is equivariant for any permutation; causal self-attention is not (reversing the rows changes which rows each token may see); adding a positional encoding to `X` before attention is not.

## Examples

```
permute_rows([[1, 1], [2, 2], [3, 3]], [2, 0, 1])       → [[3, 3], [1, 1], [2, 2]]
permute_rows([[1, 1], [2, 2]], [0, 0])                  → ValueError

is_permutation_equivariant(lambda X: 2 * X, X, perm)              → True     row-wise
is_permutation_equivariant(lambda X: np.cumsum(X, axis=0), X, perm) → False    depends on order
is_permutation_equivariant(lambda X: np.sort(X, axis=0), X, perm)   → False    invariant, not equivariant

is_permutation_equivariant(lambda X: self_attention(X, p), X, perm)               → True
is_permutation_equivariant(lambda X: self_attention(X, p, causal=True), X, [3, 2, 1, 0]) → False
is_permutation_equivariant(lambda X: self_attention(X + P, p), X, perm)           → False    P = positional encoding
```

## Constraints

- `n ≤ 16`, `d, d_k, d_v ≤ 16`. float64 numpy.
- `perm` is a `list[int]`; the identity permutation must give `True` for every `fn`.
- numpy only; no torch in the solution.

## Hints

1. Write the three steps of `self_attention` as three matrix products and one softmax. Which axis does the softmax run over, and what `keepdims` do you need so the row max broadcasts?
2. `np.triu` with `k=1` marks strictly-above-diagonal entries. How do you turn that into "set these scores to `-inf`" without a Python loop?
3. Fancy indexing `X[perm]` already reorders rows. What does it return: a view or a copy? How do you check that `perm` contains each index exactly once?
4. In the harness, which of the two sides applies `perm` to the *output*? Why would comparing `fn(X[perm])` with `fn(X)` test the wrong property?

## Explain-back

- If self-attention cannot tell positions apart, how does a transformer know that "dog bites man" differs from "man bites dog"? Name two ways position is injected.
- Why does the causal mask break equivariance? Describe what token 0 may see before and after reversing the rows.
- Is a per-token MLP (the FFN in a transformer block) permutation-equivariant? Is a convolution? Is an RNN?
- The tests reverse the rows to break causal attention. Is there *any* non-identity permutation that leaves causal attention equivariant? Argue from which rows output `i` is allowed to see before and after permuting.
