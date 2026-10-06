# Positional encodings: sinusoidal and RoPE

Topic: 4. The transformer block
Difficulty: 2 of 3

## Problem

Attention is permutation-equivariant: shuffle the input rows and the output rows shuffle with them. Position has to be injected on purpose. Write the two ways a transformer does it, in pure Python on flat lists (no numpy, no torch; `math` is allowed).

### `sinusoidal_pe(pos: int, d_model: int) -> list[float]`

The "Attention Is All You Need" encoding for **one** position. The `d_model` slots are read as `d_model / 2` **adjacent pairs**: pair `i` occupies indices `2i` (sine) and `2i + 1` (cosine), with

```
inv_freq_i = 1 / 10000 ** (2 * i / d_model)
pe[2 * i]     = sin(pos * inv_freq_i)
pe[2 * i + 1] = cos(pos * inv_freq_i)
```

So `sinusoidal_pe(0, d_model)` is `[0.0, 1.0, 0.0, 1.0, …]` — sine first, cosine second, in every pair. The result is a plain `list[float]` of length `d_model`, every entry in `[-1, 1]`. This vector is **added** to the token embedding, not concatenated to it, which is why it has exactly `d_model` entries.

### `apply_rope(vec: list[float], pos: int) -> list[float]`

Rotary position embedding. Take the same adjacent pairs of `vec` — `(vec[2i], vec[2i+1])` — and rotate each one in its own plane by the angle `pos * inv_freq_i`, using the same `inv_freq_i` as above with `d_model = len(vec)`:

```
theta = pos * inv_freq_i
out[2 * i]     = vec[2 * i] * cos(theta) - vec[2 * i + 1] * sin(theta)
out[2 * i + 1] = vec[2 * i] * sin(theta) + vec[2 * i + 1] * cos(theta)
```

RoPE is applied to `Q` and `K` only, never to `V`, and it rotates rather than adds, so it preserves the length of the vector: `|apply_rope(v, pos)| == |v|` for every `pos`, and `apply_rope(v, 0) == v`. The point of the construction is that for any query `q` at position `m` and key `k` at position `n`,

```
dot(apply_rope(q, m), apply_rope(k, n))
```

depends on `m` and `n` **only through `m - n`**: shift both positions by the same amount and the attention score does not move.

### Both functions

- Raise `ValueError` if `d_model` (or `len(vec)`) is not a positive even number, or if `pos` is negative.
- Return new lists; never mutate `vec`.
- Banned: numpy, torch, and anything from `torch.nn` — these are a dozen lines of `math`.

Tests compare floats within `1e-9`.

## Examples

```
sinusoidal_pe(0, 8)   → [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
sinusoidal_pe(1, 4)   → [0.841471, 0.540302, 0.009999833, 0.99995]
sinusoidal_pe(2, 2)   → [0.909297, -0.416147]
sinusoidal_pe(1, 3)   → ValueError                     odd d_model has no pairs

apply_rope([1.0, 0.0], 0)   → [1.0, 0.0]               position 0 is the identity rotation
apply_rope([1.0, 0.0], 1)   → [0.540302, 0.841471]     rotated by 1 radian
apply_rope([3.0, 4.0], 7)   → a vector of length 5.0

q = [1.0, 2.0, 0.5, -1.0]
dot(apply_rope(q, 5), apply_rope(q, 3)) == dot(apply_rope(q, 2), apply_rope(q, 0))     both gaps are 2
```

## Constraints

- `d_model ≤ 64`, `pos ≤ 10000`.
- Pure Python; `math` only. No numpy, no torch.
- Floats match within `1e-9`.

## Hints

1. `sinusoidal_pe(0, d)` is given to you as `[0, 1, 0, 1, …]`. Which of `sin` and `cos` is 0 at angle 0, and which is 1? What does that tell you about which index of a pair gets which function?
2. The exponent is `2 * i / d_model`, where `i` counts *pairs*, not slots. For `d_model = 8`, how many distinct frequencies are there, and what is `inv_freq` for the first and the last pair? Which pair turns fastest as `pos` grows?
3. A rotation in the plane by angle θ sends `(x, y)` to what? Write the 2×2 matrix down before you write the loop — which entry carries the minus sign, and what happens to `x² + y²`?
4. Rotate `q` by `mθ` and `k` by `nθ` in the same plane and take the dot product. If you write both as complex numbers, what angle is left over after the multiplication — and why does that make the score a function of `m - n` alone?

## Explain-back

- Positional information is added to the token embedding rather than concatenated to it. What would concatenating cost, and what would it do to `d_model` and to every weight matrix downstream?
- Without any positional encoding, what exactly does self-attention compute for "dog bites man" versus "man bites dog"? Name the property that makes those two identical.
- RoPE is applied to `Q` and `K` but not to `V`. Why does rotating the values make no sense, and what would break if you did it anyway?
- Sinusoidal encodings give an absolute position; RoPE gives a relative one. For a model asked to read a passage longer than anything it saw in training, which behaves better, and why?
