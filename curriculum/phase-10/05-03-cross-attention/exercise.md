# Cross-attention

Topic: 5. Encoder, decoder and encoder-decoder
Difficulty: 2 of 3

## Problem

In an encoder-decoder model — the original Transformer, T5, Whisper — the encoder reads the source once and the decoder consults it at every step. The sublayer that does the consulting is cross-attention, and the only thing that distinguishes it from self-attention is **where Q comes from and where K and V come from**. Get that backwards and the shapes usually still work, which is why it is worth writing once by hand.

Write, in numpy (`float64`, no batch dimension):

```
cross_attention(dec_x, enc_out, params, enc_mask=None) -> tuple[np.ndarray, np.ndarray]
```

- `dec_x` has shape `(n, d_model)`: the decoder's own hidden states, `n` of them.
- `enc_out` has shape `(m, d_model)`: the encoder's final output, `m` source positions. `n` and `m` are unrelated — a 4-token prompt can be translated into 11 tokens.
- `params` is a dict of `float64` arrays:

| key | shape |
| --- | --- |
| `"wq"` | `(d_model, d_k)` |
| `"wk"` | `(d_model, d_k)` |
| `"wv"` | `(d_model, d_v)` |
| `"wo"` | `(d_v, d_model)` |

The computation:

```
Q = dec_x   @ params["wq"]        (n, d_k)    queries come from the DECODER
K = enc_out @ params["wk"]        (m, d_k)    keys   come from the ENCODER
V = enc_out @ params["wv"]        (m, d_v)    values come from the ENCODER
scores  = Q @ K.T / sqrt(d_k)     (n, m)
weights = softmax(scores, over the last axis, i.e. over the m encoder positions)
out     = (weights @ V) @ params["wo"]        (n, d_model)
```

Return `(out, weights)`, shapes `(n, d_model)` and `(n, m)`.

**No causal mask.** The source is fully available from the first decoding step, so decoder position 0 may look at every encoder position. The causal mask belongs to the decoder's *self*-attention sublayer, which is a different sublayer.

**Decoder positions never mix here.** Row `i` of the output is built from `dec_x[i]` and `enc_out` alone, so changing `dec_x[3]` can move output row 3 and nothing else. That is the fingerprint of cross-attention, and it is how you catch yourself having written self-attention over `dec_x` by mistake.

**`enc_mask`** is `None` or a sequence of `m` booleans over the **encoder** positions, where `True` means keep and `False` means blocked (source padding). A blocked position's score becomes `-inf` **before** the softmax, so its weight comes out exactly `0.0` and the surviving weights renormalize — every row of `weights` still sums to `1.0` within `1e-12`. Zeroing weights after the softmax instead would leave the rows summing to less than 1, and the output would silently shrink toward zero.

**Errors** — raise `ValueError` if `dec_x` or `enc_out` is not 2-D, if their last dimensions differ, if `len(enc_mask) != m`, or if `enc_mask` is all `False` (a softmax over nothing is undefined).

Banned: `torch` anywhere in your solution, `nn.MultiheadAttention`, `F.scaled_dot_product_attention`. numpy only. The tests use torch to build the expected values; your solution must not.

## Examples

```
dec_x of shape (3, 8), enc_out of shape (5, 8)

out, weights = cross_attention(dec_x, enc_out, params)
out.shape      → (3, 8)      the DECODER length, not the encoder length
weights.shape  → (3, 5)      one row per decoder position, one column per encoder position
weights.sum(axis=1) → [1., 1., 1.]

cross_attention(enc_out, dec_x, params)[0].shape → (5, 8)      arguments swapped: wrong length

enc_out of shape (1, 8)
cross_attention(dec_x, enc_out, params) → weights of all ones, and every output row equal to
                                          (enc_out[0] @ wv) @ wo

enc_mask = [True, True, False, True, False]
weights[:, 2] → exactly 0.0            and the rows still sum to 1.0
cross_attention(dec_x, enc_out, params, [False] * 5) → ValueError
```

## Constraints

- `n, m` at most 16; `d_model, d_k, d_v` at most 32.
- numpy only; no torch in the solution.
- Match the PyTorch reference within `1e-10`; weight rows sum to 1 within `1e-12`.

## Hints

1. Before writing anything, label the three projections with their source: which of `Q`, `K` and `V` is built from `dec_x`, and which from `enc_out`? Then write down the shape of every intermediate from `Q` through `out`.
2. `weights` has shape `(n, m)`. Which axis does the softmax run over, and what does "the weights sum to 1" mean in words — one decoder token spreads its attention over what?
3. Where do the output's `n` rows come from, given that `V` has `m` rows? Follow the shape of `weights @ V` and say which length survives the matmul and which one is contracted away.
4. `enc_mask` marks source positions, so it blocks whole columns of the score grid, not cells of a triangle. What value must go in before the softmax to make a weight come out as 0 while the row still sums to 1 — and what is the row sum if you instead set the weight to 0 afterwards?

## Explain-back

- In cross-attention, where do Q, K and V come from? Say it as a sentence of the form "the X asks, the Y answers", and then say what changes if you swap them.
- "Encoder-decoder is obsolete, decoder-only models won." Name a current system that is encoder-decoder and say what its encoder reads that a decoder could not comfortably take as tokens.
- The decoder block has two attention sublayers. Which one is causally masked, which one is not, and why would masking the other one be a bug rather than a safety measure?
- A decoder-only model can translate perfectly well by putting the source in the prompt. What does that arrangement do instead of cross-attention, and what does the encoder-decoder get in exchange for the extra machinery?
