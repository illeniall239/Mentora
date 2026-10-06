# Incremental attention with a KV cache

Topic: 12. Inference II: KV cache, quantization and context windows
Difficulty: 3 of 3

## Problem

During decoding a transformer emits one token at a time. Recomputing attention over the whole prefix at every step is O(T²) per token; instead the model keeps the past keys and values in a **KV cache** and only projects the new token. Write that step for a single head in PyTorch.

`incremental_attention(cache, q, k, v)`:

- `cache` is a dict with exactly two entries, `"K"` and `"V"`, each a float tensor of shape `(t, d)` holding the keys and values of the `t` tokens already processed. For the first token `t = 0` (the tensors have shape `(0, d)`). The cache never holds queries.
- `q`, `k`, `v` are the new token's query, key and value, each of shape `(d,)`.
- Append `k` and `v` to the cache (as new rows at the end) to get `K_new`, `V_new` of shape `(t + 1, d)`, then compute the new token's attention output

  ```
  weights = softmax(q · K_newᵀ / √d)         shape (t + 1,), sums to 1
  out     = weights @ V_new                   shape (d,)
  ```

- Return `(out, new_cache)` where `new_cache = {"K": K_new, "V": V_new}`. Build **new** tensors (`torch.cat`); do not modify the tensors in the input `cache` in place, and do not add any other keys.

Because the new token is the last one in the sequence, it may attend to every cached position and itself, so no mask is needed: `out` must equal the **last row** of full causal attention `softmax(mask(Q Kᵀ / √d)) V` over the whole sequence, within 1e-5. The test builds a seeded sequence, runs your function token by token and compares each step against that full computation.

Allowed: matmul, `torch.softmax`, `torch.cat`, `math.sqrt`. Banned: `torch.nn.functional.scaled_dot_product_attention`, `torch.nn.MultiheadAttention`, or any built-in attention/cache class.

## Examples

```
d = 2, empty cache: K, V of shape (0, 2)
incremental_attention(cache, q=[1, 0], k=[1, 0], v=[3, 4])
    → out = [3.0, 4.0]            only one position to attend to, weight 1
      new_cache["K"] = [[1, 0]], new_cache["V"] = [[3, 4]]

cache K = [[1, 0]], V = [[3, 4]]; q = [1, 0], k = [-1, 0], v = [5, 6]
    scores = [1, -1] / √2 = [0.7071, -0.7071]  →  weights = [0.8044, 0.1956]
    → out = [3.391, 4.391], new_cache["K"] has 2 rows
```

## Constraints

- `d ≤ 32`, sequences of at most 16 tokens, float32, CPU.
- No batch and no head dimension: exactly the shapes above.
- Tolerance 1e-5.

## Hints

1. Which of Q, K and V does a past token contribute when a *new* token attends to it? Which one is only ever needed for the current token?
2. How do you stack a `(t, d)` tensor and a `(d,)` vector into `(t + 1, d)`? What must you do to the vector first?
3. Scores are `q · K_newᵀ`, shape `(t + 1,)`. Where does `√d` go, and along which dimension does the softmax run?
4. Why does the last row of a causal attention matrix need no mask at all? What does that tell you about why the cache alone is enough?

## Explain-back

- The cache stores K and V but not Q. Why is the query of a past token never needed again, while its key and value are?
- Prefill computes the whole prompt in one parallel pass; decode calls your function once per token. Which one is bound by compute and which by memory bandwidth, and why does the cache turn each decode step from O(T²) into O(T)?
- Without the cache, each new token would recompute `K` and `V` for all previous tokens. What is the total cost of generating T tokens with and without the cache?
- Multi-query and grouped-query attention share K/V across heads. What does that change in this function, and why does it shrink the cache by exactly the head ratio?
