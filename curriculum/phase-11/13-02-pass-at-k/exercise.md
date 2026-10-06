# pass@k

Topic: 13. Evaluating generative models
Difficulty: 2 of 3

## Problem

Code models are scored by whether *any* of `k` sampled attempts passes the unit tests. Sampling `k` attempts per problem and counting how often one passed is a terrible estimate: it is noisy, and it is wasteful because you cannot reuse the samples for a different `k`. The HumanEval trick is to sample `n > k` attempts once, count the `c` that pass, and work out exactly how often a random subset of `k` of them would have contained a passing one:

```
pass@k = 1 − C(n − c, k) / C(n, k)
```

which is the probability that `k` samples drawn **without replacement** from your `n` are not all failures.

Pure Python with `math` (`math.comb` is allowed and is the point).

- `pass_at_k(n: int, c: int, k: int) -> float`:
  - `c = 0` gives exactly `0.0` — no draw of failures can contain a success.
  - `n − c < k` gives exactly `1.0` — there are not enough failures to fill a draw of `k`, so every draw must contain a success. `C(n − c, k)` is 0 in that case, so the formula already says so, but say it explicitly rather than hoping your `comb` agrees.
  - `k = n` gives 1.0 whenever `c > 0`.
  - Raise `ValueError` unless `n >= 1`, `1 <= k <= n` and `0 <= c <= n`.
- `pass_at_k_from_results(results: list[bool], k: int) -> float` — the same thing from the raw per-sample outcomes: `n` is `len(results)` and `c` is the number of `True`s. Raise `ValueError` on an empty list (and propagate the checks above). The order of `results` must not matter — taking "the first `k`" is exactly the estimator this function exists to replace.

## Examples

```
pass_at_k(10, 1, 1)    → 0.1        C(9,1)/C(10,1) = 9/10
pass_at_k(10, 1, 5)    → 0.5        C(9,5)/C(10,5) = 126/252
pass_at_k(10, 1, 10)   → 1.0        only 9 failures exist, so a draw of 10 must include the success
pass_at_k(10, 5, 2)    → 0.7777...  1 − C(5,2)/C(10,2) = 1 − 10/45
pass_at_k(5, 0, 3)     → 0.0
pass_at_k(5, 5, 1)     → 1.0
pass_at_k(5, 2, 6)     → ValueError   k > n
pass_at_k(5, 6, 2)     → ValueError   c > n

pass_at_k_from_results([True, False, False, False, False, False, False, False, False, False], 5) → 0.5
pass_at_k_from_results([False]*9 + [True], 5)                                                    → 0.5
pass_at_k_from_results([], 1)                                                                    → ValueError
```

## Constraints

- `n` up to 1000. Absolute tolerance 1e-9.
- `math.comb` gives exact integers; divide only at the end so nothing overflows or loses precision.

## Hints

1. Read the formula as a probability. What does `C(n − c, k) / C(n, k)` count, and what event does subtracting it from 1 leave you with?
2. Work out `pass_at_k(10, 1, 5)` by hand. If instead you had sampled 5 attempts once and that one passing attempt was among them, what would you have reported? And if it was not?
3. What is `C(9, 10)`? Decide what your code should do before you find out what `math.comb` does with it.
4. Hold `n` and `c` fixed and raise `k` from 1 to `n`: what does the estimate do, and does it ever go down? Now hold `n` and `k` and raise `c`.

## Explain-back

- Why is this estimator "unbiased" while "sample `k`, check if any passed, average over problems" is not — if the second one is also right on average, what is actually wrong with it?
- A model scores pass@1 = 0.3 and pass@100 = 0.9. What does the gap tell you about the model, and what does it tell you about shipping it in a product where nobody runs the tests?
- pass@k needs an executable test suite. Which of the failure modes in this topic — contamination, judge bias, Goodhart — does that protect you from, and which does it leave wide open?
- A problem that the model solves 1 time in 10 scores pass@5 ≈ 0.5 here. Does your deployment see a 50% success rate? What would have to be true of the user's workflow for that number to mean anything to them?
