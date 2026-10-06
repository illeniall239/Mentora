# A tiny GPT in PyTorch

Topic: 6. The language-modelling objective and a tiny GPT
Difficulty: 3 of 3

## Problem

Build a decoder-only transformer small enough to train on a laptop in a second, then watch it overfit a repeating sequence. Everything lives in one `nn.Module`.

`TinyGPT(vocab, block, d_model, n_heads, n_layers)` has exactly these learnable parts:

- a token embedding `nn.Embedding(vocab, d_model)` and a learned position embedding `nn.Embedding(block, d_model)`, added together (not concatenated);
- `n_layers` **pre-LN** blocks, each computing `x = x + attn(ln1(x))` then `x = x + mlp(ln2(x))`, where
  - `attn` is causal multi-head self-attention written by you: one `nn.Linear(d_model, 3 * d_model)` producing Q, K and V, split into `n_heads` heads of `d_model // n_heads` dims, scores `Q Kᵀ / √d_head` with every position above the diagonal set to `−inf` **before** the softmax, then the heads concatenated and passed through an output `nn.Linear(d_model, d_model)`;
  - `mlp` is `nn.Linear(d_model, 4 * d_model)` → `nn.GELU()` → `nn.Linear(4 * d_model, d_model)`;
  - `ln1`, `ln2` are `nn.LayerNorm(d_model)`;
- a final `nn.LayerNorm(d_model)` and an untied LM head `nn.Linear(d_model, vocab)`.

Raise `ValueError` in the constructor if `d_model` is not divisible by `n_heads`.

- `forward(idx, targets=None)` takes a `(B, T)` long tensor with `T ≤ block` (raise `ValueError` otherwise) and returns `(logits, loss)`: `logits` is `(B, T, vocab)` and `loss` is `None` when `targets` is omitted, otherwise `F.cross_entropy` over all `B · T` positions of `logits` against the `(B, T)` long tensor `targets`. The caller passes targets already shifted by one; `forward` must not shift anything.
- `generate(ids, n)` takes a `(B, T)` long tensor and returns a `(B, T + n)` long tensor: `n` times, run the model on the **last `block` tokens** of the current sequence, take the `argmax` of the final position's logits, and append it. The input prefix is returned unchanged in front. Greedy decoding only; sampling comes in Topic 11. Run it under `torch.no_grad()`.

Do not use `nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer`, `nn.MultiheadAttention` or `F.scaled_dot_product_attention`: the attention is the point. `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.GELU`, `F.softmax`, `F.cross_entropy`, `masked_fill`, `torch.tril` and `view`/`transpose`/`reshape` are all fine.

The test seeds with `torch.manual_seed`, builds `TinyGPT(8, 8, 32, 2, 1)` on windows cut from a repeating sequence of period 5, and requires the initial loss to be within 0.6 of `ln 8 ≈ 2.08` and the loss after 300 full-batch AdamW steps at `lr = 3e-3` to fall below 0.1, after which `generate` must continue the pattern.

## Examples

```
model = TinyGPT(vocab=8, block=8, d_model=32, n_heads=2, n_layers=1)
logits, loss = model(torch.zeros(4, 6, dtype=torch.long))
logits.shape     → (4, 6, 8)
loss             → None
sum(p.numel() for p in model.parameters())   → 13544

x = torch.tensor([[0, 3, 5, 1, 6, 0, 3, 5]]); y = torch.tensor([[3, 5, 1, 6, 0, 3, 5, 1]])
logits, loss = model(x, y);  loss.item()     → ≈ 2.1  (near ln 8 at init)

changing x[0, 5] must not change logits[0, :5]   (causal mask)
model(torch.zeros(1, 9, dtype=torch.long))   → ValueError  (9 > block)

after training:  model.generate(torch.tensor([[0, 3]]), 5)  → [[0, 3, 5, 1, 6, 0, 3]]
```

## Constraints

- `vocab`, `block`, `d_model` at most 64; `n_layers` at most 4. float32 on CPU.
- Nothing in the module may create its own randomness besides parameter initialisation; `generate` is deterministic.

## Hints

1. Position `t` may only see positions `0 … t`. In a `(T, T)` score matrix, which entries does that forbid, and why must they be `−inf` rather than `0` before the softmax?
2. `Q`, `K`, `V` come out of one Linear as `(B, T, 3·d_model)`. Which `view` and `transpose` turn `Q` into `(B, n_heads, T, d_head)` so a batched matmul attends per head, and how do you undo it before the output projection?
3. Your loss is computed over `B · T` predictions at once even though generation runs one token at a time. Which piece of the architecture makes that parallel training legal?
4. At init, why is the loss close to `ln vocab`? After 300 steps the loss is below 0.1 on the training windows: what would it be on a sequence with a different period, and is that a problem here?

## Explain-back

- Training scores all `T` positions in one forward pass, yet `generate` needs `n` forward passes for `n` tokens. Why the asymmetry, and what makes the training-time version sound?
- Your model reached a training loss of 0.002. Is it a good language model? What is the smallest experiment that would tell you?
- What would happen to the causality test if you zeroed the attention weights *after* the softmax instead of using `−inf` before it?
- The model was never given a "write the whole sequence" objective, only "predict the next token". How does a whole continuation come out anyway, and where could an early mistake take it?
