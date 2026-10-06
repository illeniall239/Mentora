# Conv2d, valid mode (recap)

Topic: 1. From CNNs to the Vision Transformer (ViT)
Difficulty: 1 of 3

## Problem

Write `conv2d_valid(img: np.ndarray, kernel: np.ndarray) -> np.ndarray` for a single-channel image.

Slide the kernel over every position where it fits completely inside the image ("valid" mode, no padding, stride 1). At each position multiply the overlapping window by the kernel elementwise and sum. Do it the way deep-learning libraries do: cross-correlation, so the kernel is **not** flipped.

For an `H × W` image and a `kh × kw` kernel the result is `(H − kh + 1) × (W − kw + 1)`. Raise `ValueError` if the kernel is larger than the image in either dimension.

Only numpy is allowed. Write the sliding loop yourself: no `np.convolve`, `scipy.signal`, or `torch.nn.functional.conv2d`. Do not change `img` or `kernel`.

## Examples

```
img = [[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]]
kernel = [[1, 0],
          [0, 1]]
conv2d_valid(img, kernel) → [[ 6,  8],
                             [12, 14]]      1·1 + 5·1 = 6, 2 + 6 = 8, ...

conv2d_valid(img, [[1]])              → img unchanged
conv2d_valid(img, np.ones((3, 3)))    → [[45]]
```

## Constraints

- `img` is a 2-D float array up to 64 × 64; `kernel` is 2-D up to 7 × 7.
- Results are compared with `assert_allclose` at 1e-9.
- Time: O(H · W · kh · kw). Two nested loops over output positions with a vectorised window sum is fine.

## Hints

1. For output position `(i, j)`, which slice of `img` lines up with the kernel? Write its row range and column range in terms of `i`, `j`, `kh`, `kw`.
2. How many valid positions are there along the rows? Check with a 3-row image and a 2-row kernel.
3. What single numpy expression turns a window and a kernel of the same shape into one number?
4. If your result for the asymmetric kernel `[[1, 0], [0, 0]]` picks the bottom-right pixel of each window instead of the top-left, what did you do that a true convolution does but cross-correlation does not?

## Explain-back

- What two inductive biases does this operation bake in that a transformer over patches does not? Give a concrete example of each.
- Why does the output shrink, and what would you change to keep it `H × W`?
- Deep-learning "convolution" is really cross-correlation. Why does that not matter when the kernel is learned?
- A 3 × 3 kernel sees 9 pixels. After two such layers, how many input pixels influence one output value, and why does that matter for recognising a large object?
