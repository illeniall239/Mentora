# Temperature, top-k and top-p

Topic: 11. Inference I: decoding and sampling
Difficulty: 2 of 3

## Problem

A language model ends each step with a vector of logits, one per token. Before sampling, decoders reshape that distribution with three knobs. Implement them in pure Python on lists of floats. Every function returns a **new** list of probabilities that sums to 1 (never mutate the input).

- `apply_temperature(logits: list[float], T: float) -> list[float]` returns `softmax(logits / T)`. Temperature divides the **logits**, before the softmax. Subtract the max before exponentiating so large logits do not overflow. Raise `ValueError` if `T <= 0`.
- `top_k_filter(probs: list[float], k: int) -> list[float]` keeps the `k` largest probabilities, sets every other entry to `0.0`, then renormalizes the survivors so they sum to 1. Ties at the boundary: the entry with the **lower index** wins. `k >= len(probs)` returns the input unchanged (as a copy). Raise `ValueError` if `k < 1`.
- `top_p_filter(probs: list[float], p: float) -> list[float]` (nucleus sampling) sorts the probabilities in descending order (ties: lower index first), walks down accumulating them, and keeps every entry up to and **including** the one at which the cumulative probability first reaches `p` (cumulative `>= p`). Everything else becomes `0.0`; the kept entries are renormalized. `p = 1.0` keeps everything with nonzero probability. Raise `ValueError` if `p <= 0` or `p > 1`.

Positions must be preserved: the output has the same length as the input, and token `i` stays at index `i`. Renormalize **after** filtering, never before. The tests only use probabilities whose partial sums are exact in floating point (halves, quarters, eighths), so a plain `>=` comparison is enough.

## Examples

```
apply_temperature([0.0, ln 3], 1.0)   → [0.25, 0.75]
apply_temperature([0.0, ln 3], 2.0)   → [0.366..., 0.634...]     flatter than at T = 1
apply_temperature([0.0, ln 3], 0.1)   → [0.0000167..., 0.99998...] nearly greedy
apply_temperature([1000.0, 1001.0], 1.0) → [0.2689..., 0.7310...] no overflow

top_k_filter([0.1, 0.5, 0.2, 0.2], 2)   → [0.0, 0.7142..., 0.2857..., 0.0]   tie broken by index
top_k_filter([0.1, 0.5, 0.4], 3)        → [0.1, 0.5, 0.4]

top_p_filter([0.5, 0.25, 0.125, 0.125], 0.75) → [0.6666..., 0.3333..., 0.0, 0.0]   exactly p: stop here
top_p_filter([0.5, 0.25, 0.125, 0.125], 0.8)  → [0.5714..., 0.2857..., 0.1428..., 0.0]
top_p_filter([0.125, 0.5, 0.375], 0.5)        → [0.0, 1.0, 0.0]
```

## Constraints

- Lists have 1 to 1 000 entries. `probs` is a valid distribution (non-negative, sums to 1 within 1e-9).
- Pure Python with `math`. No numpy, no torch.
- Absolute tolerance in the tests: 1e-6.

## Hints

1. `softmax(x / T)` and `softmax(x) / T` are not the same thing. Which one keeps the result a distribution without renormalizing, and which one is temperature?
2. `sorted(range(len(probs)), key=...)` gives you indices, not values. How do you sort by probability descending while keeping lower indices first on ties?
3. In top-p, when you have accumulated `0.5` and `p = 0.75`, is the next entry with probability `0.25` in or out? What does "the smallest set whose mass reaches `p`" say?
4. After zeroing the losers, what do you divide by so the survivors sum to 1 again, and why can that divisor never be 0 for valid input?

## Explain-back

- A user sets `top_p = 0.9` and says "so it samples from 90% of the vocabulary". What does it actually keep, and how many tokens might that be for a peaked versus a flat distribution?
- Does temperature change what the model knows, or only how it picks? Where in the pipeline does it act, and what happens as `T → 0` and `T → ∞`?
- Decoders chain these filters, renormalizing after each. Does the order matter? Give a distribution where top-k = 2 then top-p = 0.9 keeps a different set than top-p = 0.9 then top-k = 2.
- Someone claims "temperature 0 makes the API fully deterministic". Which of your functions is deterministic, and why might a served model still vary between runs?
