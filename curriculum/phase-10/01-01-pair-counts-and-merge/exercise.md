# Pair counts and merge

Topic: 1. Tokenization and BPE
Difficulty: 1 of 3

## Problem

Byte-pair encoding is built from two small operations on a list of token IDs. Write both in pure Python (no numpy, no libraries).

- `get_pair_counts(ids: list[int]) -> dict[tuple[int, int], int]` counts every adjacent pair `(ids[i], ids[i + 1])`. Keys appear in the order their pair was first seen while scanning left to right (a plain `dict` filled in scan order does this).
- `merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]` returns a new list where every non-overlapping occurrence of `pair`, scanned left to right, is replaced by `new_id`.

Neither function may change the list passed in.

## Examples

```
get_pair_counts([1, 2, 1, 2, 3])   → {(1, 2): 2, (2, 1): 1, (2, 3): 1}
merge([1, 2, 1, 2, 3], (1, 2), 4)  → [4, 4, 3]
merge([1, 1, 1], (1, 1), 9)        → [9, 1]        left to right, no overlap
get_pair_counts([7])               → {}
```

## Constraints

- `ids` has 0 to 100 000 integers.
- Both functions run in O(n).
- Pure Python only.

## Hints

1. How many adjacent pairs does a list of length n have? What does that say about the loop bounds?
2. When you find the pair at position i, how far should the scan jump so `[1, 1, 1]` produces `[9, 1]` and not `[9, 9]`?
3. What must happen to the last element when it is not part of a matched pair?
4. Building a new list versus editing in place: which one keeps the "must not change the input" rule for free?

## Explain-back

- Why does BPE start from 256 byte IDs rather than from characters or words? What breaks if you start from Unicode code points?
- After `merge` replaces `(1, 2)` with `4`, which pair counts change and which stay the same? Why does a real trainer recount instead of patching?
- Is the tokenizer trained by gradient descent? What is the "training signal" here?
- Why must overlapping occurrences be skipped? Give an input where merging overlaps would give a list that cannot be decoded back.
