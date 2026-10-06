# The vision-to-LLM projector

Topic: 12. Multimodal models
Difficulty: 2 of 3

## Problem

A ViT gives you one feature vector per patch, 1024-wide say, and the LLM wants token embeddings 4096-wide in its own space. The bridge in LLaVA-1.5 is almost disappointingly small: a two-layer MLP applied to each patch vector independently. That is the whole "multimodal" connection — it is the LLM's own layers, trained on visual instruction data, that do the understanding.

Write the projector in numpy. Do **not** import torch in your solution; the test compares against torch itself.

`project(features: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray) -> np.ndarray`

With `features` of shape `(N, d_vision)`, `W1` `(d_vision, d_hidden)`, `b1` `(d_hidden,)`, `W2` `(d_hidden, d_llm)` and `b2` `(d_llm,)`:

```
hidden = gelu(features @ W1 + b1)          # (N, d_hidden)
out    = hidden @ W2 + b2                  # (N, d_llm)
```

Write the activation as its own exported function too — `gelu(x: np.ndarray) -> np.ndarray`, elementwise, returning the same shape as `x`.

- `gelu` is the **tanh approximation**, which is what `torch.nn.functional.gelu(x, approximate="tanh")` computes:

  ```
  gelu(x) = 0.5 · x · (1 + tanh( sqrt(2/π) · (x + 0.044715 · x³) ))
  ```

  So `gelu(1) = 0.8411919906...`, `gelu(-1) = -0.1588080093...` and `gelu(0) = 0`. Negative inputs are damped, not zeroed: a projector built with ReLU would give a different answer and the test says so.
- Each row of `features` is one patch token and is projected on its own: row `i` of the output depends only on row `i` of the input. Permuting the rows permutes the output rows and nothing else. Keep the matrices on the right of the `@` — `W1 @ features` is a different (usually ill-shaped) computation.
- Biases broadcast over the `N` axis: `b1` is added to every row of the hidden activations, not to every column.
- Raise `ValueError` if `features`, `W1` or `W2` is not 2-D, if `b1` or `b2` is not 1-D, or if any of the widths disagree: `features.shape[1] != W1.shape[0]`, `W1.shape[1] != b1.shape[0]`, `W1.shape[1] != W2.shape[0]`, `W2.shape[1] != b2.shape[0]`.

Banned: `torch`, `scipy`, and `math.erf`-based exact GELU — the tanh form above is the one being tested.

## Examples

```
features = [[1.0, -1.0]], W1 = I₂, b1 = [0, 0], W2 = I₂, b2 = [0, 0]
project(...)                     → [[0.8411919906082768, -0.1588080093917233]]
      the projector is the identity, so what you see is pure GELU
      (ReLU would give [[1.0, 0.0]])

features (4, 5), W1 (5, 7), b1 (7,), W2 (7, 3), b2 (3,)
project(...).shape               → (4, 3)

W1 = zeros, b1 = [1.0, -1.0]: every row of hidden is gelu([1, -1]), so every output row is identical
project(features, W1, b1, W2, b2)[i] == gelu(b1) @ W2 + b2   for every i

features (4, 5) with W1 (6, 7)   → ValueError
```

## Constraints

- All arrays are `float64`. `N` up to 1024, widths up to 64.
- numpy only. Absolute tolerance 1e-10 against the torch reference.

## Hints

1. Write down the shape of every intermediate before you write any code. Which axis of `features` must line up with which axis of `W1` for the matmul to mean "project each patch"?
2. `features @ W1` is `(N, d_hidden)` and `b1` is `(d_hidden,)`. What does numpy broadcasting do with that addition, and what would it do if `b1` had shape `(N,)` instead?
3. The GELU formula has `x³` in it. What does that term do for large positive `x`, for large negative `x`, and near 0 — and why would ReLU change the output of a *trained* projector rather than just being "close enough"?
4. If you accidentally wrote `W1.T @ features.T`, what shape would come out, and would any of the shape checks catch it?

## Explain-back

- The projector is two matmuls and a nonlinearity, with no attention and no mixing between patches. Where does the actual image understanding happen, and what does that say about what visual instruction tuning is training?
- Why project at all? What goes wrong if you feed raw 1024-wide ViT features into an LLM whose embeddings are 4096-wide and have a learned scale?
- LLaVA-1.0 used a single linear layer and 1.5 used this two-layer MLP. What can the MLP represent that the linear map cannot, and would you expect the gain to be large?
- Cross-attention fusion (Flamingo) keeps the image features outside the token sequence. Compare the two designs on sequence length, on how many LLM parameters must be touched, and on how easily a new vision encoder is swapped in.
