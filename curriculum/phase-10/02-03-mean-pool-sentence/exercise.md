# Mean-pool a sentence embedding

Topic: 2. Embeddings revisited: static, contextual and tied
Difficulty: 2 of 3

## Problem

A retrieval model turns a sentence into one vector by averaging the hidden states of its real tokens, then L2-normalizing so that a dot product between two sentence vectors is a cosine. Sentences in a batch are padded to the same length, and the hidden states at padded positions are garbage that must not leak into the average.

Write `mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor` in PyTorch with tensor ops (no loops over the batch or over positions):

- `hidden` has shape `(B, T, D)`; `mask` has shape `(B, T)` and is `1`/`True` at real tokens and `0`/`False` at padding. Accept a bool, integer or float mask.
- For each sequence, average the hidden states of the positions where `mask` is on (sum of those states divided by their **count**, not by `T`), then divide the result by its L2 norm. Return shape `(B, D)`.
- Two copies of the same sentence padded to different lengths must give the same vector (within `1e-6`), whatever the hidden states at the padded positions contain.
- Raise `ValueError` if any sequence has no real token (its mask row is all zero), or if `hidden` and `mask` disagree on `B` or `T`.
- The result must be differentiable with respect to `hidden`.

Do not use `torch.nn.functional.normalize` for the division; write it out.

## Examples

```
hidden = [[[3., 0.], [1., 2.], [99., 99.]]]        B = 1, T = 3, D = 2
mask   = [[1, 1, 0]]
mean_pool(hidden, mask)     → [[0.8944, 0.4472]]     mean of the first two = [2, 1], then / sqrt(5)

mask   = [[1, 0, 0]]
mean_pool(hidden, mask)     → [[1., 0.]]

mask   = [[0, 0, 0]]
mean_pool(hidden, mask)     → ValueError
```

## Constraints

- `B ≤ 16`, `T ≤ 64`, `D ≤ 128`. float32.
- Results are compared within `1e-6`; every output row has norm 1 within `1e-5`.
- PyTorch on CPU, tensor ops only.

## Hints

1. `hidden` is `(B, T, D)` and `mask` is `(B, T)`. What must you do to `mask` so that broadcasting multiplies every feature of a padded position by 0?
2. After masking, which `dim` do you sum over to collapse the positions, and what is the matching per-sequence divisor? Why is it not `T`?
3. `torch.norm` and `keepdim=True`: what shape must the norms have so that `(B, D) / norms` divides each row by its own length?
4. Which of the two failure conditions can you detect from `mask.sum(dim=1)` alone?

## Explain-back

- Why does dividing by `T` instead of the token count make a sentence's vector depend on how long the *other* sentences in its batch are?
- After L2-normalization, what does the dot product of two sentence vectors equal? Why do retrieval systems want that?
- Is a sentence embedding just the average of the static token embeddings? Which layer's states are pooled here, and why does that matter for "bank" in two different sentences?
- These pooled vectors are trained contrastively for retrieval. Would pooling a GPT's hidden states, trained only for next-token prediction, give good retrieval vectors? What is missing?
