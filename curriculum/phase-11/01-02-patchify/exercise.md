# Patchify

Topic: 1. From CNNs to the Vision Transformer (ViT)
Difficulty: 2 of 3

## Problem

A Vision Transformer does not read pixels one at a time. It cuts the image into `p × p` squares, flattens each square into one vector, and treats those vectors as the token sequence. Write the three plain-Python functions that do that on nested lists (no numpy in this exercise).

- `patchify(img: list[list[list[float]]], p: int) -> list[list[float]]` — `img` is `H × W × C` (`img[row][col][channel]`). Return `N` patch vectors, each of length `p · p · C`. Patches are ordered like reading a page: left to right along the top row of patches, then the next row. Inside a patch, flatten by row, then column, then channel, i.e. the order of `for r: for c: for ch: img[r][c][ch]`.
- `unpatchify(patches: list[list[float]], h: int, w: int, p: int, c: int) -> list[list[list[float]]]` — the exact inverse, so `unpatchify(patchify(img, p), H, W, p, C)` returns `img`.
- `num_patch_tokens(h: int, w: int, p: int, cls: bool) -> int` — the number of tokens the encoder sees: `(H / p) · (W / p)`, plus one if `cls` is true.

All three raise `ValueError` if `H` or `W` is not a multiple of `p`. Do not change the inputs.

## Examples

```
img = [[[1], [2], [3], [4]],
       [[5], [6], [7], [8]]]            H = 2, W = 4, C = 1
patchify(img, 2)                        → [[1, 2, 5, 6], [3, 4, 7, 8]]

img2 = [[[1, 10], [2, 20]],
        [[3, 30], [4, 40]]]             H = 2, W = 2, C = 2
patchify(img2, 2)                       → [[1, 10, 2, 20, 3, 30, 4, 40]]

num_patch_tokens(224, 224, 16, True)    → 197
num_patch_tokens(224, 224, 16, False)   → 196
num_patch_tokens(225, 224, 16, False)   → ValueError
```

## Constraints

- `H`, `W` up to 64; `C` from 1 to 4; `p` from 1 to `min(H, W)`.
- Pure Python lists in, pure Python lists out. Values are compared exactly.
- Time: O(H · W · C).

## Hints

1. How many patches are there along the width, and given a patch index `n`, which patch row and patch column is it? Think integer division and modulo.
2. For patch row `pr` and patch column `pc`, which pixel rows and columns does the patch cover?
3. Write the flattening as three nested loops in the order the problem gives. What would change if you swapped the inner two?
4. For `unpatchify`, every element of a patch vector has a position `k`; how do you recover `(r, c, ch)` from `k` using `p` and `C`?

## Explain-back

- A 224 × 224 image with `p = 16` gives 196 tokens. Why does a 225 × 225 image not simply give "a bit more than 196", and what do real ViT pipelines do about it?
- Is a patch vector a pixel? What information does the encoder lose, and what does it keep, compared with a CNN's first layer?
- Why does the order of patches matter to the model at all, given that self-attention is permutation-invariant?
- Halving `p` does what to the token count, and what does that do to the cost of self-attention?
