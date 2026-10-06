# Encode by rank

Topic: 1. Tokenization and BPE
Difficulty: 2 of 3

## Problem

A trained BPE tokenizer is an **ordered** list of merges. Encoding new text must replay them in the order they were learned (their *rank*), not in the order pairs happen to appear in the text. Pure Python only.

Write `encode(text: str, merges: dict[tuple[int, int], int]) -> list[int]`:

- Start from the UTF-8 bytes of `text` (IDs 0–255).
- `merges` maps a pair of IDs to the new ID it became. Its rank is its position in the dict (insertion order; `dict` preserves it). For a tokenizer trained the usual way this is also the order of the new IDs (256, 257, …), but your code must go by the dict's order.
- Repeat: among all adjacent pairs currently in the sequence that appear in `merges`, pick the one with the **lowest rank** and replace every non-overlapping occurrence of it (scanning left to right) with its new ID. Stop when no adjacent pair is in `merges`.
- `merges` may be empty, and `text` may be empty. Never modify `merges`.

You will want a `merge(ids, pair, new_id)` helper like the one from `01-01`; you may copy it into this file. Do not import from other exercises.

## Examples

```
merges = {(98, 99): 256, (97, 256): 257, (97, 98): 258}      ranks 0, 1, 2
encode("abc", merges)   → [257]         b,c merge first (rank 0), then a,256 (rank 1)
                                        left-to-right would merge a,b first and give [258, 99]
encode("ab", merges)    → [258]
encode("xyz", merges)   → [120, 121, 122]
encode("", merges)      → []
```

## Constraints

- `text` is at most 5 000 characters; `merges` has at most 512 entries.
- Any UTF-8 text is valid input, including emoji.
- Pure Python only.

## Hints

1. At each step you may have several mergeable pairs in the sequence. Which one did the *trainer* apply first, and why must the encoder copy that order?
2. How do you turn "position in the dict" into a number you can take a `min` over? What should a pair that is not in `merges` score, so it is never chosen?
3. After merging the lowest-rank pair everywhere, can a *lower*-rank pair newly appear? What does the answer tell you about whether you need to rescan from the top?
4. Can a merge whose halves are themselves merged tokens (such as `(256, 256)`) ever fire before the merges that create `256`? What guarantees this in your loop?

## Explain-back

- Two encoders give different tokens for the same text: one applies merges by rank, the other merges the first pair it meets scanning left to right. Which one matches what the model saw during training, and why does that matter for the embedding table?
- Why is the merge list ordered at all? What would break if the trainer stored merges as an unordered set?
- Is the tokenizer learned by gradient descent? What is the objective the trainer optimizes, and where does the text's statistics enter?
- If you concatenate two separately encoded strings, is the result the same as encoding the concatenation? Give a case where it is not.
