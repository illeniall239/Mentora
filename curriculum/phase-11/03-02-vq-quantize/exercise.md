# Vector quantization against a codebook

Topic: 3. Autoencoders, VAEs and VQ-VAE
Difficulty: 2 of 3

## Problem

A VQ-VAE replaces the continuous latent with a lookup: every encoder output is snapped to the nearest entry of a learned codebook, and what travels onward is an **integer index**. That is the step that turns an image or a second of audio into discrete tokens a language model can consume. Write the snap.

`vq_quantize(vectors: np.ndarray, codebook: np.ndarray) -> tuple[np.ndarray, np.ndarray]`

- `vectors` is a 2-D `float64` array of shape `(N, D)`: `N` encoder outputs.
- `codebook` is a 2-D `float64` array of shape `(K, D)`: the `K` learned entries, in index order.

Return `(indices, quantized)`:

- `indices`: an integer array of shape `(N,)`, where `indices[i]` is the index of the codebook entry **closest in Euclidean distance** to `vectors[i]`. Closest in distance, not largest in dot product — those disagree the moment the codebook entries have different lengths. Comparing squared distances is fine and avoids a square root.
- `quantized`: a `float64` array of shape `(N, D)` with `quantized[i] = codebook[indices[i]]`, an exact copy of one codebook row. Nothing is averaged, blended or interpolated: the output of this function can only ever contain the `K` vectors you were given, which is the whole point — `quantized` carries no more information than `indices` does.

Ties go to the **smallest** index, so the function is deterministic and two runs on the same inputs give the same tokens.

A vector that is already a codebook entry quantizes to itself with zero error. Adding entries to a codebook can never make the total squared error worse: if `codebook_big` starts with all the rows of `codebook_small`, the error under `codebook_big` is less than or equal to the error under `codebook_small`.

Do not modify the inputs. Raise `ValueError` if either argument is not 2-D, if their widths disagree, or if either has zero rows.

## Examples

```
cb = array([[0., 0.], [1., 1.], [5., 5.]])
vq_quantize(array([[0.1, 0.1], [0.9, 1.2], [4., 4.]]), cb)
                      → (array([0, 1, 2]), array([[0., 0.], [1., 1.], [5., 5.]]))

vq_quantize(cb, cb)   → (array([0, 1, 2]), cb)        entries quantize to themselves

vq_quantize(array([[1., 0.]]), array([[1., 0.], [5., 0.]]))
                      → (array([0]), array([[1., 0.]]))
                        distance 0 vs 16 picks entry 0, though the dot product 1 vs 5 prefers entry 1

vq_quantize(array([[0., 0.]]), array([[1., 0.], [-1., 0.]]))
                      → (array([0]), array([[1., 0.]]))   tie, smallest index

vq_quantize(array([[0., 0.]]), zeros((0, 2)))   → ValueError
vq_quantize(array([[0., 0.]]), zeros((3, 5)))   → ValueError
```

## Constraints

- `N` up to 4096, `K` up to 512, `D` up to 64.
- numpy only; no `scipy`, no `torch`, no explicit Python loop over `N * K` pairs (vectorize or broadcast).
- Distances compared within `1e-9`; `quantized` must match the codebook rows exactly.

## Hints

1. You need every pairwise distance between `N` vectors and `K` entries. What shape is that table, and which axis do you then take the argmin over?
2. Broadcasting `(N, 1, D)` against `(1, K, D)` gives differences of shape `(N, K, D)`. Which axis do you square and sum to collapse it to `(N, K)`?
3. Does taking the square root change which entry is closest? What does skipping it save you?
4. Once you have `indices`, how much work is `quantized`? What does the answer tell you about how much information the index carries?

## Explain-back

- After quantization the decoder only ever sees one of `K` vectors. What has been thrown away compared with a VAE's continuous latent, and what has been gained?
- The argmin is not differentiable. How does VQ-VAE train the encoder through it, and what would happen without that trick?
- "Audio tokens are like text tokens, one per word." Using this function, say what one token actually corresponds to and how many you would need for a second of audio.
- If one codebook entry is never the nearest for any input, what happens to it during training, and what is that failure called?
