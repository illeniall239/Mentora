# Recall@k for image-text retrieval

Topic: 2. Contrastive learning and CLIP
Difficulty: 2 of 3

## Problem

A CLIP batch of `N` pairs gives an `N x N` similarity matrix whose diagonal holds the true pairs and whose off-diagonal entries are the in-batch negatives. Retrieval is scored with recall@k: how often the true partner lands in the top `k` of the ranked list. There are two lists, and they are not the same list.

`recall_at_k(sim_matrix: np.ndarray, k: int) -> tuple[float, float]`

- `sim_matrix` is a 2-D `float64` array of shape `(N, N)`, where `sim_matrix[i, j]` is the similarity between **image `i`** and **text `j`**. The correct partner of image `i` is text `i`, i.e. the diagonal.
- Return `(r_i2t, r_t2i)` in that order:
  - `r_i2t` (image to text): for each **row** `i`, rank all `N` texts by descending `sim_matrix[i, :]`; count a hit when text `i` is among the first `k`.
  - `r_t2i` (text to image): for each **column** `j`, rank all `N` images by descending `sim_matrix[:, j]`; count a hit when image `j` is among the first `k`.
- Each value is `hits / N`, a `float` in `[0, 1]`.

The matrix is not symmetric, so the two numbers generally differ: being a row's best text does not make you that text's best image. Report both.

Ties rank by index, lowest index first — exactly what a stable descending sort gives you. So if the true entry ties with a distractor that sits at a lower index, the distractor is ranked above it, and with `k = 1` the true entry misses.

`recall_at_k` with `k = N` is always `(1.0, 1.0)`, and the value never decreases as `k` grows. Do not modify the input. Raise `ValueError` if `sim_matrix` is not 2-D and square, if `N == 0`, or if `k` is not an integer in `[1, N]`.

## Examples

```
sim = [[0.9, 0.8, 0.7],
       [0.9, 0.5, 0.6],
       [0.2, 0.1, 0.3]]

recall_at_k(sim, 1)   → (0.6666666666666666, 0.3333333333333333)
recall_at_k(sim, 2)   → (0.6666666666666666, 0.6666666666666666)
recall_at_k(sim, 3)   → (1.0, 1.0)

sim = [[1.0, 1.0],
       [0.0, 1.0]]
recall_at_k(sim, 1)   → (1.0, 0.5)      column 1 ties, and index 0 wins the tie

recall_at_k(eye(4), 1)      → (1.0, 1.0)
recall_at_k(eye(4), 0)      → ValueError
recall_at_k(eye(4), 5)      → ValueError
recall_at_k(zeros((2, 3)), 1) → ValueError
```

## Constraints

- `N` up to 512.
- numpy only. A loop over the `N` rows is fine; so is a vectorized rank count.
- Values are compared within `1e-12`.

## Hints

1. "Is the true item in the top `k`" is the same question as "is the rank of the true item less than `k`". How many entries have to beat it for it to fall out of the top `k`?
2. For row `i`, which entry is the true one, and how do you count how many entries in that row are strictly greater than it without sorting anything?
3. The tie rule says a lower index wins. For an entry equal to the true one, when does it push the true one down a place and when does it not?
4. You wrote the row version. What is the smallest change that turns it into the column version, and why can you not simply reuse the row answer?

## Explain-back

- Why does CLIP retrieval report two recall numbers rather than one? Give a concrete case where image-to-text is high and text-to-image is low.
- Recall@1 on a batch of 8 and recall@1 on a pool of 5000 are both "recall@1". Which is the harder number, and why does that make batch size part of the training signal as well as the evaluation?
- The negatives here are the other items in the batch, not curated hard negatives. What does that buy, and what does it cost?
- If every similarity in the matrix were multiplied by 10, which of these numbers would change? What does that tell you about what recall@k measures compared with the contrastive loss itself?
