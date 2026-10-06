# KV-cache and MoE arithmetic

Topic: 12. Inference II: KV cache, quantization and context windows
Difficulty: 1 of 3

## Problem

Two back-of-the-envelope numbers every LLM engineer computes before choosing a GPU. Pure Python, integers in and out.

- `kv_cache_bytes(layers: int, kv_heads: int, head_dim: int, seq_len: int, batch: int, bytes_per: int) -> int` returns the bytes needed to cache the past **keys and values** (not queries) for every layer, for `batch` sequences of length `seq_len`, with `bytes_per` bytes per element (2 for fp16/bf16, 4 for fp32, 1 for int8):

  ```
  bytes = 2 × layers × kv_heads × head_dim × seq_len × batch × bytes_per
  ```

  The leading 2 is for K and V. `kv_heads` is the number of **key/value** heads: with multi-head attention it equals the query-head count; with grouped-query attention (GQA) it is smaller. Raise `ValueError` if any argument is less than 1.

- `moe_active_params(shared: int, per_expert: int, n_experts: int, top_k: int) -> int` returns the number of parameters that take part in **one token's** forward pass in a mixture-of-experts model: the `shared` parameters (attention, embeddings, norms, router) plus the `top_k` experts that the router picks, each of `per_expert` parameters:

  ```
  active = shared + top_k × per_expert
  ```

  Raise `ValueError` if `top_k < 1`, `top_k > n_experts`, `n_experts < 1`, or `shared < 0`, or `per_expert < 0`. (The total stored parameter count is `shared + n_experts × per_expert`; the tests compute that themselves to compare.)

## Examples

```
# Llama-2 7B: 32 layers, 32 KV heads, head_dim 128, 4096 tokens, batch 1, fp16
kv_cache_bytes(32, 32, 128, 4096, 1, 2)   → 2147483648      exactly 2 GiB
# Llama-3 8B uses GQA with 8 KV heads: 4× smaller
kv_cache_bytes(32, 8, 128, 4096, 1, 2)    → 536870912       512 MiB
kv_cache_bytes(1, 1, 1, 1, 1, 1)          → 2
kv_cache_bytes(32, 8, 128, 0, 1, 2)       → ValueError

# Mixtral-style: 8 experts, 2 active per token
moe_active_params(1_300_000_000, 5_600_000_000, 8, 2)  → 12_500_000_000    (total 46.1B)
moe_active_params(100, 10, 4, 4)                       → 140               dense: every expert active
moe_active_params(100, 10, 4, 5)                       → ValueError
```

## Constraints

- All arguments are Python ints; results are exact ints (no floats, no rounding).
- Pure Python.

## Hints

1. Which tensors does a decode step need from the past: Q, K, V, or only some of them? How many of those tensors are stored per layer?
2. For one token in one layer, how many numbers is one head's key? And with `kv_heads` heads? Multiply out the rest.
3. In GQA, which dimension of the formula shrinks, and why does the query-head count not appear at all?
4. In an MoE, which parameters run for *every* token and which run only when the router picks them? What is the difference between "stored" and "used"?

## Explain-back

- Why does a long context blow up memory at decode time even though the weights do not change size? Which term in the formula grows?
- Prefill processes the prompt in parallel; decode emits one token at a time. Which one is compute-bound and which is memory-bandwidth-bound, and what does the KV cache do for the second?
- An MoE with 400B total parameters and 2 of 16 experts active per token: how much compute per token, and how much memory to hold it? Why is "an MoE is cheap" only half true?
- Does the KV cache store the queries? What would you gain and what would you lose if it did?
