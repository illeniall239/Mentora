# LoRA forward pass and merge

Topic: 8. Fine-tuning, instruction tuning and LoRA
Difficulty: 2 of 3

## Problem

Implement a LoRA adapter around a frozen linear layer in numpy, and show that merging the adapter into the weights gives the same outputs with no extra work at inference.

Conventions: the layer computes `x @ W` with `x` of shape `(n, d_in)` and `W` of shape `(d_in, d_out)`. The adapter is `B` of shape `(d_in, r)` and `A` of shape `(r, d_out)`, so `B @ A` has the shape of `W`. The update is scaled by `alpha / r`.

- `init_lora(d_in: int, d_out: int, r: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]` returns `(A, B)` at their training-start values: `A = rng.standard_normal((r, d_out)) * 0.02` and `B = np.zeros((d_in, r))`. That order of draws is part of the contract.
- `lora_forward(x: np.ndarray, W: np.ndarray, A: np.ndarray, B: np.ndarray, alpha: float, r: int) -> np.ndarray` returns `x @ W + (alpha / r) · (x @ B) @ A`, which equals `x @ (W + (alpha / r) · B @ A)`. Compute it as two thin matmuls through the rank-`r` bottleneck, never by forming the `(d_in, d_out)` product `B @ A`: that is the unmerged form used during training.
- `merge_lora(W: np.ndarray, A: np.ndarray, B: np.ndarray, alpha: float, r: int) -> np.ndarray` returns the single matrix `W + (alpha / r) · B @ A`, so that `x @ merge_lora(...)` equals `lora_forward(...)` for every `x`. Do not modify `W` in place.

All three raise `ValueError` if `r < 1`, or if `B.shape[1]` or `A.shape[0]` differs from `r`, or if the shapes of `W`, `A`, `B` do not chain (`B.shape[0] != W.shape[0]` or `A.shape[1] != W.shape[1]`). Tolerance for all comparisons is `1e-6`.

## Examples

```
W = np.eye(2);  B = [[1.0], [0.0]];  A = [[0.0, 1.0]];  x = [[1.0, 2.0]]
alpha = 8, r = 1  →  scale 8
lora_forward(x, W, A, B, 8, 1)   → [[1.0, 2.0]] + 8 · ([[1.0]] @ [[0.0, 1.0]]) = [[1.0, 10.0]]
merge_lora(W, A, B, 8, 1)        → [[1.0, 8.0], [0.0, 1.0]]
alpha = 16, r = 1  →  scale 16:  lora_forward → [[1.0, 18.0]]

A, B = init_lora(4, 6, 2, np.random.default_rng(0))
A.shape → (2, 6),  B.shape → (4, 2),  B is all zeros
lora_forward(x, W, A, B, 4, 2) == x @ W      at init, exactly
```

## Constraints

- `d_in`, `d_out` at most 64; `r` at most 8; float64 numpy.
- numpy only: no torch.

## Hints

1. Group the parentheses two ways: `(x @ B) @ A` and `x @ (B @ A)`. What shapes are the intermediate results, and which grouping stays small when `d_in = d_out = 4096` and `r = 8`?
2. If `B` starts at zero, what does `B @ A` equal at step 0, and why does that matter for a fine-tune that must begin exactly at the pretrained model?
3. Where exactly does `alpha / r` multiply? If you dropped it, by what factor would the example output change?
4. What is the rank of `B @ A` at most, whatever `A` and `B` contain? Which numpy function could confirm that on `merge_lora(...) − W`?

## Explain-back

- Why does the merged model run at exactly the pretrained model's speed at inference, while an unmerged adapter costs two extra matmuls per layer?
- A colleague sets `alpha = r` "to keep things simple", then doubles `r` without touching `alpha`. What just happened to the effective learning rate of the update?
- Why must `B` (not `A`) be the zero-initialised factor? What would go wrong if both were random, and what if both were zero?
- LoRA changes `W` by a rank-8 matrix. What kind of behaviour change is that well suited to, and why is it a poor tool for injecting a large body of new facts?
