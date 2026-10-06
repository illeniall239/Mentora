# Expected calibration error

Topic: 13. Hallucination, mechanically
Difficulty: 2 of 3

## Problem

A model is **calibrated** when, among all the answers it gave with confidence 0.8, about 80% are right. Expected calibration error (ECE) measures the gap. Pure Python.

`expected_calibration_error(confidences: list[float], correct: list[bool], n_bins: int) -> float`:

- `confidences[i]` in `[0, 1]` is the model's stated probability that answer `i` is right; `correct[i]` says whether it was.
- Split `[0, 1]` into `n_bins` equal-width bins that are **open on the left and closed on the right**: bin `b` covers `(b / n_bins, (b + 1) / n_bins]`. A confidence of exactly `0.0` goes into bin 0. So with 4 bins, `0.5` belongs to `(0.25, 0.5]`, not to the bin above it, and `1.0` belongs to the last bin.
- For each non-empty bin `B` compute `acc(B)` = fraction of its answers that are correct and `conf(B)` = mean of its confidences. Then

  ```
  ECE = Σ_B  (|B| / N) · |acc(B) − conf(B)|
  ```

  Empty bins contribute nothing. A perfectly calibrated model gives 0; a model that is always 90% confident and always wrong gives 0.9.
- Raise `ValueError` if the two lists differ in length, are empty, if `n_bins < 1`, or if any confidence is outside `[0, 1]`.

## Examples

```
expected_calibration_error([0.5, 0.5, 0.5, 0.5], [True, False, True, False], 10)   → 0.0
expected_calibration_error([0.9, 0.9, 0.9], [False, False, False], 10)              → 0.9
expected_calibration_error([0.2, 0.2], [True, True], 10)                            → 0.8   underconfident
expected_calibration_error([0.8, 0.8, 0.8, 0.2], [False, False, False, False], 2)   → 0.65  (3/4)·0.8 + (1/4)·0.2
expected_calibration_error([0.5, 0.5, 0.6, 0.6], [True, True, False, False], 4)     → 0.55  0.5 and 0.6 land in different bins
expected_calibration_error([0.3], [True], 0)                                        → ValueError
```

## Constraints

- At most 10 000 answers; `n_bins` at most 100.
- Pure Python with `math`. Absolute tolerance 1e-9.

## Hints

1. For a confidence `c` and `n_bins`, which arithmetic gives the bin index so that `c = 0.5` with 4 bins lands in bin 1, `c = 0.51` in bin 2 and `c = 1.0` in bin 3? Try `math.ceil(c * n_bins) - 1` on those values, then ask what happens at `c = 0`.
2. A bin needs three numbers: how many answers, how many were right, and the sum of confidences. Which container makes that easy to accumulate in one pass?
3. Why is each bin's gap weighted by `|B| / N` rather than averaged evenly? Which of the examples above would change if you dropped the weight?
4. Confidences of exactly 0.5 all correct and exactly 0.6 all wrong: why does putting them in one bin hide the miscalibration, and what does that say about the choice of `n_bins`?

## Explain-back

- A calibrated model with ECE 0 can still be wrong 30% of the time. What does calibration promise, and what does it not?
- How does ECE relate to hallucination? If a model says "I'm 95% sure" on a made-up citation, which of the training, decoding or data causes could make it that overconfident?
- Reinforcement learning from human feedback tends to make models more confident. Which bins would you expect to move, and how would you test that with this function?
- Why can a high sequence probability (or a high stated confidence) not be read as evidence that the answer is true? Connect it to the objective the model was trained on.
