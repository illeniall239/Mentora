# Constrained decoding and repetition penalty

Topic: 11. Inference I: decoding and sampling
Difficulty: 2 of 3

## Problem

Two more things a decoder does to the logits before the softmax: forbid tokens that would break a required format (JSON, a grammar, a fixed set of answers) and discourage tokens that were already emitted. Pure Python on lists of floats; return **new** lists, never mutate the input.

- `constrained_mask(logits: list[float], allowed_ids: set[int]) -> list[float]` returns a copy of `logits` where every index **not** in `allowed_ids` is `-math.inf` and every allowed index keeps its value. After a softmax, forbidden tokens have probability exactly 0 and so can never be sampled. Raise `ValueError` if `allowed_ids` is empty or contains an index outside `0 … len(logits) − 1`.
- `repetition_penalty(logits: list[float], history: list[int], penalty: float) -> list[float]` applies the CTRL-style penalty to every token id that appears in `history` (each id penalized **once**, however often it appears): a positive logit is **divided** by `penalty`, a negative or zero logit is **multiplied** by `penalty`. Both moves push the token's probability down. Tokens not in `history` are unchanged; `penalty = 1.0` is the identity. Raise `ValueError` if `penalty <= 0`. History ids outside the vocabulary are ignored.

Also provide `softmax(xs: list[float]) -> list[float]` (max-subtracted, treating `-inf` entries as probability 0) so the tests can check what the masked logits mean; it must not raise on a row containing `-inf` as long as at least one entry is finite.

## Examples

```
constrained_mask([1.0, 2.0, 3.0], {0, 2})       → [1.0, -inf, 3.0]
softmax([1.0, -inf, 3.0])                       → [0.1192..., 0.0, 0.8807...]
constrained_mask([1.0, 2.0], set())             → ValueError
constrained_mask([1.0, 2.0], {5})               → ValueError

repetition_penalty([2.0, -1.0, 0.5], [0, 1], 2.0)   → [1.0, -2.0, 0.5]
repetition_penalty([2.0, -1.0, 0.5], [1, 1, 1], 2.0) → [2.0, -2.0, 0.5]    penalized once
repetition_penalty([2.0, -1.0, 0.5], [], 2.0)      → [2.0, -1.0, 0.5]
repetition_penalty([2.0, -1.0, 0.5], [0], 1.0)     → [2.0, -1.0, 0.5]
```

## Constraints

- Logit lists have at most 1 000 entries; `history` at most 10 000 ids.
- Pure Python with `math`.
- Absolute tolerance 1e-6.

## Hints

1. What does `math.exp(-math.inf)` return, and what does that make the softmax probability of a masked token? Why is that better than setting the probability to 0 after the softmax?
2. If you divided a negative logit by `penalty > 1`, would that token become more or less likely? What operation makes a negative logit *more* negative?
3. `history` is a list with repeats. Which built-in turns it into "each id once" before you loop?
4. Which inputs make the mask meaningless (nothing allowed) or wrong (an id past the end of the list), and where should you check them?

## Explain-back

- Constrained decoding guarantees valid JSON. Does it guarantee the values inside the JSON are true? What is the difference between shaping *form* and shaping *content*?
- Why does the repetition penalty divide positive logits but multiply negative ones? What would go wrong with "always subtract 1"?
- A frequency penalty subtracts `alpha × count` for each token; a presence penalty subtracts a constant once. When would you pick one over the other, and how does each relate to yours?
- Where would a stop sequence be checked: on logits, on sampled ids, or on decoded text? Why can a stop string split across two tokens be a problem?
