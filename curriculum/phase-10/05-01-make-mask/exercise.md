# Three attention masks

Topic: 5. Encoder, decoder and encoder-decoder
Difficulty: 1 of 3

## Problem

The mask is what separates BERT from GPT. Same block, same weights shapes, same attention maths — different boolean grid. Write `make_mask(n: int, kind: str, prefix_len: int | None = None) -> list[list[bool]]`, returning the `n` × `n` grid for one of the three families.

**Convention, used everywhere below: `mask[i][j] is True` means query `i` is allowed to attend to key `j` (keep), and `False` means it is blocked.** The opposite convention exists in the wild — PyTorch's `attn_mask` for `nn.MultiheadAttention` blocks on `True` — so read every API before you trust it. Here, `True` = keep.

- `"bidirectional"` (BERT, the encoder): every position sees every position. The whole grid is `True`.
- `"causal"` (GPT, the decoder): position `i` sees positions `0` through `i` **inclusive**. `mask[i][j]` is `True` exactly when `j <= i`. The diagonal is kept: a token always sees itself, so row 0 is not all-`False` — an all-blocked row would make its softmax undefined. This is the lower triangle *including* the diagonal, so row `i` has `i + 1` `True` entries.
- `"prefix"` (prefix-LM, the middle ground: T5-style, and what a chat model's prompt behaves like): the first `prefix_len` positions are read bidirectionally, everything from `prefix_len` on is generated causally. Every position may see the whole prefix, and otherwise only its own past: `mask[i][j]` is `True` exactly when `j < prefix_len` **or** `j <= i`.

Rules:
- Return a fresh `list` of `n` `list`s of `n` Python `bool`s (`True` / `False`, not `1` / `0`, not numpy scalars). Rows must be independent objects: mutating `m[0]` must not change `m[1]`.
- Raise `ValueError` if `n < 1`, if `kind` is not one of the three strings, if `kind == "prefix"` and `prefix_len` is `None`, or if `prefix_len` is given for `"prefix"` and is not in `0 … n`.
- `prefix_len` is ignored for the other two kinds.
- Pure Python. No numpy, no torch.

Two edges worth knowing by heart: `prefix_len=0` makes `"prefix"` identical to `"causal"`, and `prefix_len=n` makes it identical to `"bidirectional"`.

## Examples

```
make_mask(3, "bidirectional") → [[True,  True,  True ],
                                 [True,  True,  True ],
                                 [True,  True,  True ]]

make_mask(4, "causal")        → [[True,  False, False, False],
                                 [True,  True,  False, False],
                                 [True,  True,  True,  False],
                                 [True,  True,  True,  True ]]

make_mask(4, "prefix", 2)     → [[True,  True,  False, False],
                                 [True,  True,  False, False],
                                 [True,  True,  True,  False],
                                 [True,  True,  True,  True ]]

make_mask(4, "prefix", 0)     == make_mask(4, "causal")
make_mask(4, "prefix", 4)     == make_mask(4, "bidirectional")
make_mask(4, "prefix")        → ValueError
make_mask(4, "masked-lm")     → ValueError
```

## Constraints

- `1 <= n <= 256`.
- Pure Python; lists of `bool`.
- Exact grids, no tolerance.

## Hints

1. Write the three grids for `n = 3` on paper first. For each one, what is the condition on `i` and `j` that decides a single cell? Each kind is one boolean expression.
2. In the causal grid, how many `True` entries does row `i` have? If your row 0 came out empty, which comparison did you use — `<` or `<=`?
3. The prefix grid is two conditions joined by one word. Which word — `and` or `or` — and what does picking the wrong one do to the very first row?
4. Check your prefix grid at both extremes, `prefix_len = 0` and `prefix_len = n`. If either does not collapse onto one of the other two kinds, which part of your condition is off?

## Explain-back

- You are asked to build a sentiment classifier and a chatbot. Which mask does each use, what objective is each trained on, and which one can generate text one token at a time?
- Someone says "BERT can generate text like GPT, just slower". Which property of the bidirectional mask makes left-to-right generation unsound, and what does BERT actually produce at a masked slot?
- The causal mask lets a transformer train on every position of a sequence in one forward pass. Spell out why that would be unsafe without it — what would position `i` learn to predict?
- In the prefix mask, a prompt token at position 0 can see a prompt token at position 5. Why is that legitimate for a prompt but not for the generated continuation?
