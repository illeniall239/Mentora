# Interleaving image slots

Topic: 12. Multimodal models
Difficulty: 2 of 3

## Problem

A LLaVA-style model does not hand the LLM an image. It builds **one** sequence of token ids in which the image occupies a block of reserved placeholder positions, and just before the forward pass it overwrites the embeddings at those positions with the projected patch features. The placeholders are what give the image a *place*: they decide what the image attends to, what attends to it, and where it sits relative to "What is in this picture?".

Build that sequence. Pure Python, no numpy.

`interleave(text_ids: list[int], image_slots: list[int], image_token_id: int, n_per_image: int) -> tuple[list[int], list[tuple[int, int]]]`

- `image_slots[k]` is an index **into `text_ids`**: image `k`'s block goes immediately before `text_ids[image_slots[k]]`. A slot equal to `len(text_ids)` puts the block after all the text.
- Each image contributes exactly `n_per_image` copies of `image_token_id`.
- Return `(seq, spans)`. `spans[k]` is the half-open range `(start, end)` of image `k`'s block **in `seq`**, so `seq[start:end]` is `n_per_image` copies of `image_token_id` and `end - start == n_per_image`.
- Images are processed in the order given. Two images in the same slot produce two blocks back to back, in list order, with no text between them.
- Raise `ValueError` if `image_slots` is not non-decreasing, if any slot is outside `0 … len(text_ids)`, if `n_per_image < 1`, or if `image_token_id` already occurs in `text_ids` (the placeholder id is reserved; if real text could carry it, you could no longer tell which positions to overwrite).
- Do not modify the inputs.

Two invariants the tests check: dropping every `image_token_id` from `seq` gives `text_ids` back unchanged, and the text token originally at index `i` ends up at index `i + n_per_image · (number of slots ≤ i)`.

## Examples

```
interleave([5, 6, 7], [1], -200, 4)
    → ([5, -200, -200, -200, -200, 6, 7], [(1, 5)])

interleave([5, 6, 7], [0, 3], -200, 2)
    → ([-200, -200, 5, 6, 7, -200, -200], [(0, 2), (5, 7)])

interleave([5, 6], [1, 1], -200, 1)
    → ([5, -200, -200, 6], [(1, 2), (2, 3)])

interleave([], [0], -200, 3)      → ([-200, -200, -200], [(0, 3)])
interleave([5, 6, 7], [], -200, 2) → ([5, 6, 7], [])

interleave([5, 6, 7], [3, 1], -200, 2)     → ValueError   slots not in order
interleave([5, 6, 7], [4], -200, 2)        → ValueError   slot past the end of the text
interleave([5, -200, 7], [1], -200, 2)     → ValueError   placeholder id already in the text
```

## Constraints

- Up to 10 000 text ids and 32 images, `n_per_image` up to 2500. Ids are plain `int`s, compared exactly.
- Time: O(len(seq)).

## Hints

1. Walk the text from left to right and keep one index into `image_slots`. At each text position, how do you know whether one or more blocks has to come out first?
2. `start` for a block is simply "how long is `seq` right now". Which other value do you then not have to compute separately?
3. Two images share a slot. If your loop emits at most one block per text position, what happens to the second one — and does your answer still satisfy the "drop the placeholders" invariant?
4. What is left to emit once the text runs out, and which slot value corresponds to that case?

## Explain-back

- The placeholders are overwritten by projected patch features before the transformer runs. Why keep them in the id sequence at all instead of splicing the features in directly?
- Why do the spans matter to the caller? Name one thing it has to do with them that the sequence alone would not support.
- "Multimodal just means calling a captioner and pasting the caption in." What can the model do with tokens in these slots that it could not do with a caption string, and what does it lose compared with the caption?
- An image placed after the question instead of before it changes the answer. Which property of a causal decoder makes position matter here, and would a cross-attention (Flamingo-style) fusion have the same sensitivity?
