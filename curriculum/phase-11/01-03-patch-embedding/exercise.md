# Patch embedding

Topic: 1. From CNNs to the Vision Transformer (ViT)
Difficulty: 3 of 3

## Problem

Patchifying gives raw pixel vectors. A ViT does not feed those to the encoder: it projects each patch to the model width `d` with one shared linear layer, puts a learned `[CLS]` token in front, and adds learned positional embeddings. Write that whole input stage in numpy.

`patch_embed(img: np.ndarray, p: int, W: np.ndarray, pos: np.ndarray, cls_vec: np.ndarray) -> np.ndarray`

- `img` is a 3-D `float64` array of shape `(height, width, channels)`, indexed `img[row, col, channel]`.
- `p` is the patch size. Both `height` and `width` must be exact multiples of `p`.
- `W` is the shared patch-projection matrix, shape `(p * p * channels, d)`. There is **no bias**.
- `pos` is the learned positional embedding table, shape `(N + 1, d)`, where `N = (height / p) * (width / p)`. Row 0 belongs to the `[CLS]` token, row `i + 1` to patch `i`.
- `cls_vec` is the learned `[CLS]` vector, shape `(d,)`.

Build the result in this exact order:

1. Cut the image into `N` patches. Patches are ordered like reading a page: left to right along the top row of patches, then the next row of patches. Inside one patch, flatten in the order of `for r: for c: for ch: img[r, c, ch]`, giving a vector of length `p * p * channels`.
2. Project every patch with the **same** `W`: `tokens[i] = patch[i] @ W`, shape `(N, d)`.
3. Stack `cls_vec` on top as row 0, giving `(N + 1, d)`.
4. Add `pos` elementwise to the whole `(N + 1, d)` matrix, the `[CLS]` row included.

Return the `(N + 1, d)` `float64` array. Positions are **added**, never concatenated, so the width stays `d`. The projection is shared across patches — the same `W` for patch 0 and patch 195 — and it is the only thing in this stage that knows anything about pixels; everything after it is a plain transformer encoder on `N + 1` tokens.

Do not modify any input. Raise `ValueError` if `img` is not 3-D, if `p < 1`, if `height` or `width` is not a multiple of `p`, or if any of `W`, `pos`, `cls_vec` does not have the shape above.

## Examples

```
img.shape = (4, 4, 3), p = 2, W.shape = (12, 8)    → N = 4,   result.shape == (5, 8)
img.shape = (224, 224, 3), p = 16, W.shape = (768, 768)
                                                   → N = 196, result.shape == (197, 768)
W all zeros                                        → rows 1.. equal pos[1:], row 0 equals cls_vec + pos[0]
pos all zeros and cls_vec all zeros                → rows 1.. are exactly the projected patches
img.shape = (5, 4, 3), p = 2                       → ValueError
W.shape = (11, 8) with p = 2, channels = 3         → ValueError
pos.shape = (4, 8) with N = 4                      → ValueError   (N + 1 rows, not N)
```

## Constraints

- `height`, `width` at most 64; `channels` 1 to 4; `d` at most 64.
- numpy only. No `torch`, no `nn.Conv2d`, no `np.lib.stride_tricks`.
- Values are compared within `1e-12`.

## Hints

1. A patch is a vector of length `p * p * channels` and a token is a vector of length `d`. Which single matrix turns one into the other, and how many such matrices does a ViT hold for 196 patches?
2. If you collect the patches into one `(N, p * p * channels)` matrix first, how many matrix multiplications does step 2 need?
3. `pos` has `N + 1` rows but there are only `N` patches. Which row is left over, and what would the model lose if you added `pos[0]` to patch 0 instead?
4. Shapes `(1, d)` and `(N, d)` stack into `(N + 1, d)` — which numpy call does that, and in which argument order so that `[CLS]` comes first? What would broadcasting silently do if you added a `(d,)` vector instead of the `(N + 1, d)` table?

## Explain-back

- A 224x224x3 image with `p = 16` and `d = 768` becomes a `(197, 768)` matrix. Where did the 196 come from, and what does a 225x225 image do to that count?
- The patch projection is one linear layer shared by every patch. What does a CNN's first conv layer give you that this does not, and why is that difference the reason ViT needs far more pretraining data?
- At layer 1, how much of the image does a single token know about? Where does that change, and through which operation?
- Positions are added rather than concatenated, and they are learned rather than fixed. What breaks if you drop them entirely, given that self-attention cannot tell which token came from which corner?
