# Sample by inverse CDF

Topic: 11. Inference I: decoding and sampling
Difficulty: 1 of 3

## Problem

Once the distribution over the next token is fixed, a decoder draws one token from it. Implement that draw with the inverse-CDF method on a plain Python list, using a seeded `random.Random` so the result is reproducible.

`sample(probs: list[float], rng: random.Random) -> int`:

- Call `rng.random()` **exactly once** to get `u` in `[0, 1)`.
- Walk the list accumulating probabilities. Return the **first** index `i` whose cumulative probability `probs[0] + … + probs[i]` is strictly greater than `u`.
- If rounding leaves `u` above every partial sum (a sum like 0.9999999999), return the **last index with a nonzero probability**.
- Raise `ValueError` if `probs` is empty, contains a negative entry, or does not sum to 1 within `1e-6`.

Do not use `rng.choices`, `random.choices`, `numpy` or `torch`: the point is to see where the randomness enters and why a seed makes it repeatable. Any object with a `.random()` method may be passed as `rng`; the tests use a stub to pin `u`.

## Examples

```
rng = random.Random(0)
sample([0.5, 0.5], rng)             → 1        (u = 0.844…, cumulative after index 0 is 0.5 ≤ u)
sample([0.0, 1.0, 0.0], any rng)    → 1        always
sample([0.2, 0.3, 0.5], rng with u = 0.2)  → 1  (0.2 is not > 0.2, 0.5 is)
sample([0.2, 0.3, 0.5], rng with u = 0.0)  → 0
sample([0.5, 0.5, 0.0], rng with u = 0.9999999999999) → 1   never the trailing zero
sample([], rng)                     → ValueError
sample([0.6, 0.6], rng)             → ValueError
```

## Constraints

- `probs` has at most 1 000 entries.
- Pure Python with `random`, `math`.
- Over 10 000 draws with `random.Random(1234)` the empirical frequency of each token must be within 0.02 of its probability.

## Hints

1. If you drew `u` uniformly from `[0, 1)` and the first token has probability 0.3, which values of `u` should map to it? What about the second token with probability 0.5?
2. Why must the comparison be `cumulative > u` rather than `>=`? Think about `u = 0.0` and a leading probability of 0.
3. What happens to your loop if the probabilities sum to 0.9999999999 and `u = 0.99999999995`? What should be returned instead of falling off the end?
4. Where does the one call to `rng.random()` belong so that two `Random(seed)` objects replay the same tokens?

## Explain-back

- Why does the same seed give the same sample here, and why can the same seed still give different text from a served model (batching, kernels, non-deterministic reductions)?
- A token with probability 0.001 was sampled and the answer went wrong from there. Is that a bug in `sample`? What does it say about hallucination under sampling?
- How does `sample` combine with `top_k_filter` / `top_p_filter` / `apply_temperature` from the previous exercise? Which of them run before it, and does `sample` need to know they ran?
- Greedy decoding is `argmax`. Which `u` values would make `sample` behave like greedy, and why can you not get greedy by picking a "special seed"?
