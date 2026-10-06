# BERT's masked-LM corruption

Topic: 5. Encoder, decoder and encoder-decoder
Difficulty: 2 of 3

## Problem

GPT gets its training signal for free: shift the sequence by one. An encoder sees the whole sentence at once, so there is nothing to predict unless you break the input on purpose. That is the masked-LM objective, and this is the corruption step that produces it.

Write

```
mask_tokens(ids, rate, mask_id, rng, vocab_size=1000, special_ids=(0,)) -> tuple[list[int], list[int]]
```

with `ids: list[int]`, `rate: float`, `mask_id: int`, `rng: random.Random`, `vocab_size: int`, `special_ids: tuple[int, ...]`.

Return `(corrupted_ids, labels)`, both the same length as `ids`.

**Selection.** Walk the positions left to right. A position whose id is in `special_ids` is **never** selected — `special_ids` holds `[PAD]`, `[CLS]`, `[SEP]` and friends, and asking the model to reconstruct its own padding teaches it nothing while inflating the loss. Every other position is selected independently with probability `rate`, via `rng.random() < rate`.

**Corruption of a selected position**, the 80/10/10 rule from the BERT paper — do **not** always write `[MASK]`:

- with probability 0.8 the id becomes `mask_id`;
- with probability 0.1 it becomes a uniformly random id, `rng.randrange(vocab_size)`;
- with probability 0.1 it is **left exactly as it was**.

The last two exist because `[MASK]` never appears at fine-tuning or inference time. If the model only ever had to fix up `[MASK]` slots, it would learn to ignore every position that is not one; the random and unchanged tenth force it to build a usable representation of *every* token.

**Labels.** `labels[i]` is the **original** `ids[i]` when position `i` was selected, and `-100` otherwise. `-100` is the ignore index: the loss is computed only where a label exists, which is roughly `rate` of the sequence — and that is why masked-LM pretraining is less sample-efficient per token than next-token prediction, which gets a label at every position. Note what the label does *not* depend on: a position left unchanged by the 10% branch still carries its label, and that is the only way the model can be scored on it.

**Rules.**
- `ids` is never mutated, and the returned list is a new object even when nothing is selected.
- Unselected positions come back with their id untouched.
- Raise `ValueError` if `rate` is outside `0.0 … 1.0`, if `vocab_size < 1`, if `mask_id` is outside `range(vocab_size)`, or if any entry of `ids` is outside `range(vocab_size)`.
- Pure Python and the `random` module. Do not reseed `rng` and do not use the module-level `random.random()`; every draw comes from the `rng` you were handed, so the same seed gives the same corruption.

## Examples

```
rng = random.Random(0)
ids = [7, 8, 0, 9, 0, 11]                       # 0 is [PAD], special_ids=(0,)

mask_tokens(ids, 0.0, 4, rng)   → ([7, 8, 0, 9, 0, 11], [-100, -100, -100, -100, -100, -100])
mask_tokens(ids, 1.0, 4, rng)   → (corrupted,           [7,    8,    -100, 9,    -100, 11  ])
                                                         ^ the pads stayed out of it entirely
                                   labels are exact at rate 1.0; `corrupted` is usually
                                   [4, 4, 0, 4, 0, 4] but about 1 selected id in 5 is a random
                                   token or the original instead — it depends on rng.
                                   corrupted[2] == 0 and corrupted[4] == 0, always.

over 4000 non-special ids at rate=0.15: about 600 labels; of those about 480 became mask_id,
about 60 became a random id, about 60 were left alone.

mask_tokens([7], 1.5, 4, random.Random(0))      → ValueError
mask_tokens([7, 2000], 0.1, 4, random.Random(0))→ ValueError    2000 is outside range(1000)
```

## Constraints

- `len(ids)` up to 20000; `vocab_size` up to 50000.
- Pure Python; `random.Random` only. No numpy, no torch.
- Tests check proportions with wide tolerances, not exact sequences — any ordering of your `rng` calls is fine.

## Hints

1. Two things have to come out of one pass: the corrupted ids and the labels. What is the default value of a label, and at what moment in the loop do you know a position deserves a real one?
2. The 80/10/10 split is three outcomes from one draw. If you call `rng.random()` once per selected position, which intervals of `[0, 1)` do the three branches own — and what goes wrong if you draw a fresh number for each branch?
3. A position in the "leave it unchanged" tenth looks identical in the input to a position that was never selected. What single piece of the output distinguishes them, and why would the model learn nothing from that branch without it?
4. A `[PAD]` token is in `special_ids`. At which step must you check for it — before deciding whether to select, or after? What would a label on a pad position do to the loss?

## Explain-back

- Why does BERT need to corrupt its input at all, when GPT does not? Trace it back to the mask each one uses.
- 80/10/10: what specifically would the model learn to do if the split were 100/0/0, and what happens at inference time, when there is no `[MASK]` token anywhere in the input?
- A sequence of 4000 tokens at `rate=0.15` gives about 600 training signals; a decoder-only model on the same 4000 tokens gets how many? What does that imply about the pretraining compute each needs?
- Someone proposes raising `rate` to 0.9 "so the model learns more per sequence". What breaks?
