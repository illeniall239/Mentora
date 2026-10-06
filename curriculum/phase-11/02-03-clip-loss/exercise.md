# The symmetric CLIP loss

Topic: 2. Contrastive learning and CLIP
Difficulty: 3 of 3

## Problem

Write CLIP's training objective in numpy: InfoNCE in both directions over one batch of paired embeddings.

`clip_loss(img_embs: np.ndarray, txt_embs: np.ndarray, temperature: float) -> float`

- `img_embs` and `txt_embs` are 2-D `float64` arrays of shape `(N, d)`, straight out of the two projection heads and **not** normalized. Row `i` of each is the same pair, so the correct partner of image `i` is text `i`: the positives are the diagonal and every off-diagonal entry is a free in-batch negative.

Steps, in order:

1. **L2-normalize every row** of both matrices, so each embedding has unit length. Without this the dot products are dominated by vector length rather than direction, and the temperature stops meaning anything.
2. Similarity matrix: `logits[i, j] = (img_n[i] . txt_n[j]) / temperature`, shape `(N, N)`. The temperature **divides** — a small temperature sharpens the softmax, a large one flattens it.
3. Cross-entropy in both directions against the labels `[0, 1, ..., N - 1]`:
   - `loss_i2t` = mean over rows `i` of `-log(softmax(logits[i, :])[i])`,
   - `loss_t2i` = mean over columns `j` of `-log(softmax(logits[:, j])[j])`.
4. Return `0.5 * (loss_i2t + loss_t2i)` as a Python `float`.

Both directions are needed: softmax over a row normalizes against the other texts, softmax over a column normalizes against the other images, and the matrix is not symmetric, so the two terms differ. The `0.5` keeps the scale comparable to a single cross-entropy.

Properties your implementation must have:

- Scaling `img_embs`, `txt_embs` or any single row of either by a positive constant leaves the loss unchanged.
- `N = 1` gives exactly `0.0` whatever the embeddings are: with no negatives there is nothing to tell apart.
- With `N = 2` and `img_embs = txt_embs = [[1, 0], [0, 1]]`, the loss is exactly `log(1 + exp(-1 / temperature))`.
- Shuffling `txt_embs` so the pairs no longer match raises the loss.

Use a numerically stable softmax (subtract the row or column maximum before exponentiating). Do not modify the inputs. Raise `ValueError` if either array is not 2-D, if their shapes differ, if `N == 0`, if `temperature <= 0`, or if any row has zero norm.

## Examples

```
I = array([[1., 0.], [0., 1.]])
clip_loss(I, I, 1.0)        → 0.31326168751822286     == log(1 + exp(-1))
clip_loss(I, I, 0.5)        → 0.12692801104297263     == log(1 + exp(-2))
clip_loss(I, I, 0.05)       → 2.06e-09                sharper softmax, smaller loss
clip_loss(10 * I, I, 1.0)   → 0.31326168751822286     normalization kills the scale
clip_loss(I, I[::-1], 1.0)  → 1.3132616875182228      shuffled pairs, far worse

clip_loss(array([[1., 0.]]), array([[0., 1.]]), 1.0)   → 0.0      N = 1, no negatives
clip_loss(I, I, 0.0)                                   → ValueError
clip_loss(I, zeros((2, 2)), 1.0)                       → ValueError
```

## Constraints

- `N` up to 256, `d` up to 128.
- numpy only; no `torch`, no `scipy`, no library cross-entropy or softmax. The tests use torch to build the expected values; your solution must not.
- Compared within `1e-10`.

## Hints

1. Row `i` of the similarity matrix is a classification problem over `N` classes. Which class is the right answer, and what does that make the cross-entropy target vector?
2. `-log(softmax(z)[i])` can be written as `logsumexp(z) - z[i]`. Which of the two forms is less likely to overflow when `temperature` is small, and what do you subtract before the `exp` either way?
3. The second direction uses the same matrix. Which axis does its softmax run over, and is `logits.T` with the same row-wise code the same computation?
4. Multiplying every embedding by 3 must leave the loss alone, but multiplying `temperature` by 3 must not. Which step makes the first true, and what happens to the gap between the diagonal and the off-diagonal logits in the second?

## Explain-back

- Where do the negatives come from in this loss, and what happens to the difficulty of the task when the batch grows from 8 to 32768?
- The temperature divides the similarities. Describe the loss surface in the two limits, temperature near zero and temperature very large, and say what the model learns in each.
- CLIP is trained on raw web pairs rather than labelled classes. Which part of this loss stands in for the labels?
- If you ran only the image-to-text half, what failure mode would show up in retrieval? Build a 2x2 similarity matrix on which the two halves disagree.
