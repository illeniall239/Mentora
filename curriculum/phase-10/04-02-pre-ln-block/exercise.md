# A pre-LN decoder block in numpy

Topic: 4. The transformer block
Difficulty: 3 of 3

## Problem

Build one decoder block of a modern LLM in numpy, piece by piece, and match a PyTorch block loaded with the same weights. Everything is a single sequence: `x` has shape `(T, d_model)` and dtype `float64`. There is no batch dimension and no dropout (inference only).

Banned: `torch` anywhere in your solution, `nn.MultiheadAttention`, `F.scaled_dot_product_attention`, `nn.TransformerEncoderLayer` / `TransformerDecoderLayer`, and any library normalization layer. Use numpy only. The tests use torch to build the expected values; your solution must not.

### `rms_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray`

RMSNorm, with **no learned gain** (the gain lives in `params` and is applied by the caller):

```
rms_norm(x) = x / sqrt(mean(x ** 2, over the last axis) + eps)
```

Note what it does *not* do: it never subtracts the mean, and it never touches any axis but the last. Each token row is normalized entirely on its own, so adding, removing or reordering rows cannot change another row's output. It is scale-invariant: `rms_norm(c * x) == rms_norm(x)` for any `c > 0`, up to `eps`. A row of zeros comes back as zeros, not `nan`. The shape is unchanged.

### `multi_head_attention(x, params, n_heads, causal) -> np.ndarray`

`multi_head_attention(x: np.ndarray, params: dict, n_heads: int, causal: bool = True) -> np.ndarray`, where `d_k = d_model // n_heads`:

1. `Q = x @ params["wq"]`, `K = x @ params["wk"]`, `V = x @ params["wv"]`, each `(T, d_model)`. There are no biases anywhere in this block.
2. Split each into `n_heads` heads by **reshaping the last axis**: `(T, d_model)` to `(T, n_heads, d_k)`, so head `h` owns columns `h * d_k` through `(h + 1) * d_k - 1`. Heads are *not* interleaved.
3. Each head attends on its own: `softmax(Q_h K_h^T / sqrt(d_k)) V_h`. If `causal`, set the scores at `j > i` to `-inf` **before** the softmax, so row `i` sees keys `0 … i` inclusive: the diagonal is kept.
4. Concatenate the heads back in the same column order, `(T, n_heads, d_k)` to `(T, d_model)`, then apply the output projection: `@ params["wo"]`.

Returns shape `(T, d_model)`. Changing `n_heads` changes the result but never the shape and never the parameter count. Raise `ValueError` if `n_heads` does not divide `d_model`, if `n_heads < 1`, or if `x` is not 2-D.

### `transformer_block(x: np.ndarray, params: dict) -> np.ndarray`

The **pre-LN** block: normalization sits *inside* each residual branch, and the residual stream itself is never normalized.

```
h   = x + multi_head_attention(rms_norm(x) * params["ln1_g"], params, params["n_heads"], causal=True)
out = h + ffn(rms_norm(h) * params["ln2_g"])

ffn(z) = gelu(z @ params["w1"]) @ params["w2"]
```

The block is always causal. There is no normalization after the second residual add: in a real model a single final norm sits after the whole stack, not inside the block. `gelu` is the tanh approximation, matching `torch.nn.functional.gelu(..., approximate="tanh")`:

```
gelu(z) = 0.5 * z * (1 + tanh(sqrt(2 / pi) * (z + 0.044715 * z ** 3)))
```

A direct consequence of pre-LN: if `params["wo"]` and `params["w2"]` are all zeros, both sublayers contribute nothing and `transformer_block(x, params)` returns `x` **exactly**. Post-LN would not.

### `params`

A dict of `float64` arrays plus one int:

| key | shape | role |
| --- | --- | --- |
| `"n_heads"` | `int` | heads for the attention sublayer |
| `"ln1_g"` | `(d_model,)` | RMSNorm gain before attention |
| `"wq"`, `"wk"`, `"wv"`, `"wo"` | `(d_model, d_model)` | attention projections |
| `"ln2_g"` | `(d_model,)` | RMSNorm gain before the FFN |
| `"w1"` | `(d_model, d_ff)` | FFN up-projection |
| `"w2"` | `(d_ff, d_model)` | FFN down-projection |

Tests match a PyTorch block built from the same arrays within `1e-10`.

## Examples

```
rms_norm(array([[1., 2., 3.]]))     → [[0.46291, 0.92582, 1.38873]]    the mean is 0.92582, not 0
rms_norm(array([[10., 20., 30.]]))  → the same row                     scale-invariant
rms_norm(array([[0., 0., 0.]]))     → [[0., 0., 0.]]                   eps, not nan
rms_norm(array([[5., 5.]]))         → [[1., 1.]]

multi_head_attention(x_of_shape_4_by_8, params, 3, True)  → ValueError   3 does not divide 8
multi_head_attention(x, params, n_heads, causal=True)[0]  == (x[0] @ wv) @ wo   row 0 sees only itself

params["wo"][:] = 0; params["w2"][:] = 0
transformer_block(x, params)        → x                                 pre-LN, exactly
```

## Constraints

- `T` at most 16, `d_model` at most 32, `d_ff = 4 * d_model`, `n_heads` in `{1, 2, 4, 8}`.
- numpy only; no torch in the solution.
- Match the PyTorch block within `1e-10`.

## Hints

1. LayerNorm subtracts the mean and divides by the standard deviation; RMSNorm skips the first half. Over which axis is `mean(x ** 2)` taken, and what does `keepdims=True` buy you in the division that follows?
2. You have `Q` of shape `(T, d_model)` and want `(n_heads, T, d_k)` so one matmul does every head at once. Which `reshape` gives you `(T, n_heads, d_k)`, and which `transpose` reorders it? What has to happen on the way back so that the round trip is an identity?
3. Once `d_model` is split, is the softmax scale `sqrt(d_model)` or `sqrt(d_k)` — and why does the answer follow from the length of the vectors actually being dotted? Which score entries become `-inf`, and is row `i` allowed to see key `i`?
4. Write the two residual lines for pre-LN and for post-LN side by side. In which of them is the value flowing along the residual stream ever rewritten by a norm, and what does each one return when both sublayer outputs are identically zero?

## Explain-back

- Doubling `n_heads` at a fixed `d_model`: what changes in the computation, and what happens to the number of parameters in the block? Why?
- The FFN runs on every token. Can token 3's FFN output depend on token 1? Which single sublayer in this block moves information between positions?
- RMSNorm's statistics are taken over which axis? If you put a second, wildly different sequence into the same batch, which outputs change — and what is the answer to the same question for BatchNorm?
- Pre-LN trains stably without a learning-rate warmup where post-LN often does not. Looking at the two residual equations, what does a gradient travelling from `out` back to `x` pass through in each?
