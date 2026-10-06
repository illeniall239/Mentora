# LoRA parameter count

Topic: 8. Fine-tuning, instruction tuning and LoRA
Difficulty: 1 of 3

## Problem

LoRA freezes a weight matrix `W` of shape `(d_in, d_out)` and learns a low-rank update `ΔW = B A` with `B` of shape `(d_in, r)` and `A` of shape `(r, d_out)`. Count what that saves. Pure Python only.

- `full_param_count(d_in: int, d_out: int) -> int` returns the number of entries in `W`, `d_in · d_out`.
- `lora_param_count(d_in: int, d_out: int, r: int) -> int` returns the number of **trainable** parameters LoRA adds for that matrix: the entries of `B` plus the entries of `A`, `r · (d_in + d_out)`. `W` itself is frozen and not counted.

Both raise `ValueError` if `d_in` or `d_out` is less than 1, and `lora_param_count` also raises it if `r < 1` or `r > min(d_in, d_out)` (a rank that large is no longer "low").

## Examples

```
full_param_count(4096, 4096)          → 16777216
lora_param_count(4096, 4096, 8)       → 65536       0.39 % of the full matrix
lora_param_count(4096, 11008, 8)      → 120832      an FFN projection is not square
lora_param_count(4096, 4096, 16)      → 131072      twice the rank, twice the parameters
lora_param_count(4096, 4096, 5000)    → ValueError
```

## Constraints

- Dimensions are up to 65 536; results are exact integers.
- Pure Python only.

## Hints

1. How many numbers are in a `(d_in, r)` matrix, and how many in an `(r, d_out)` matrix? Which of the two grows with `d_out`?
2. Does `ΔW = B A` have the same shape as `W`? What does that say about whether the product needs storing separately at inference?
3. If `r` equals `min(d_in, d_out)`, how does `r · (d_in + d_out)` compare with `d_in · d_out`? Is that still a saving?
4. Which inputs should each function reject, and should `full_param_count` know anything about `r`?

## Explain-back

- The full matrix has 16.8 M entries; LoRA trains 65 k. Where do the other parameters go during fine-tuning: are they deleted, frozen, or still used in the forward pass?
- After training, `W + B A` can be added into a single matrix. What does that mean for latency at inference compared with the original model?
- Would you fine-tune (LoRA or full) to teach a model that your company's CEO changed last week? What tends to happen to hallucination when you fine-tune on new facts, and what would you do instead?
- In SFT, which tokens of a (prompt, response) pair should contribute to the loss, and why does including the prompt teach the wrong thing?
