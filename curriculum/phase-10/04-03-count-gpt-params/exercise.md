# Counting the parameters of a GPT

Topic: 4. The transformer block
Difficulty: 2 of 3

## Problem

Write `count_params(vocab, d_model, n_layers, n_heads, ff_mult, tied, n_positions=1024) -> int`, returning the exact number of **learned scalars** in a GPT-2-style decoder-only model. No tensors are built: this is arithmetic, and the answer for GPT-2 small must land on the published 124M.

Count exactly these, and nothing else:

**Embeddings**
- token embedding: `vocab * d_model`
- learned absolute position embedding: `n_positions * d_model`

**Each of the `n_layers` blocks** (`d_ff = ff_mult * d_model`)
- LayerNorm before attention: `2 * d_model` (a gain **and** a bias)
- Q, K and V projections: `3 * (d_model * d_model + d_model)` — weight **and** bias for each
- output projection `W_O`: `d_model * d_model + d_model` — it is a real projection with real parameters, not a free concatenation
- LayerNorm before the FFN: `2 * d_model`
- FFN up: `d_model * d_ff + d_ff`; FFN down: `d_ff * d_model + d_model`

**After the stack**
- final LayerNorm: `2 * d_model`
- the output head (unembedding), `vocab * d_model`, with **no** bias — but only when `tied` is `False`. When `tied` is `True` the head reuses the token embedding matrix, so those scalars are already counted and must not be counted a second time.

`n_heads` appears in the signature and changes nothing in the total. It only has to be legal: raise `ValueError` if it does not divide `d_model`. Also raise `ValueError` if any of `vocab`, `d_model`, `n_layers`, `n_heads`, `ff_mult` or `n_positions` is not a positive integer (`n_layers` may be 0, for an embeddings-only model).

Return a plain `int`, exact — no floats, no rounding, no approximation.

## Examples

```
count_params(50257, 768, 12, 12, 4, True)             → 124439808        GPT-2 small
count_params(50257, 768, 12, 1,  4, True)             → 124439808        heads are free
count_params(50257, 768, 12, 12, 4, False)            → 163037184        untied head: + 50257 * 768
count_params(10, 4, 1, 2, 4, True, n_positions=5)     → 312
count_params(10, 4, 0, 2, 4, True, n_positions=5)     → 68               40 + 20 + 8
count_params(50257, 768, 12, 5, 4, True)              → ValueError       5 does not divide 768
```

The 312 above, written out: token `10*4 = 40`, positions `5*4 = 20`, one block `8 + 60 + 20 + 8 + 80 + 68 = 244`, final LayerNorm `8`.

## Constraints

- `vocab` up to 300000, `d_model` up to 16384, `n_layers` up to 200, `n_positions` up to 2000000.
- Pure Python integers; no numpy, no torch, no floating point anywhere in the arithmetic.
- Exact equality, not a tolerance.

## Hints

1. Write the per-block count as a formula in `d_model` and `d_ff` before you write any code. How many `d_model * d_model` blocks of weights does the attention sublayer hold — and does the concatenation of the heads need any of them?
2. GPT-2 uses `nn.Linear`, which has a weight *and* a bias. For a layer of shape `(d_model, d_ff)`, how many bias scalars are there, and is it `d_model` or `d_ff`?
3. Where does `n_heads` enter the shapes of `W_Q`, `W_K`, `W_V` and `W_O`? If `d_k = d_model / n_heads` and there are `n_heads` heads, what is `n_heads * d_k`?
4. With `tied=True` the same matrix is used twice: once as a lookup and once as the output projection. How many distinct learned numbers does a matrix used twice contribute, and what does `tied=False` change about that?

## Explain-back

- Someone says "we doubled the heads, so the model got bigger". What actually changed, and what did not?
- In GPT-2 small, which holds more parameters: all twelve attention sublayers together, or all twelve FFNs together? Give the ratio from your formula and say where it comes from.
- Why does the parameter count not depend on the sequence length actually used at inference — and which single term in your formula does depend on the *maximum* context?
- A model tying its input embedding to its output head saves `vocab * d_model` parameters. What has to be true about those two matrices for that to be sound, and what shape constraint does it force?
