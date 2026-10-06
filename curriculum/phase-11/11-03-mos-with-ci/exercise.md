# MOS with a confidence interval

Topic: 11. Text-to-speech
Difficulty: 2 of 3

## Problem

TTS quality is still judged by people: listeners rate utterances 1 (bad) to 5 (excellent) and the system is scored by the **mean opinion score**. A bare MOS of 4.1 says nothing on its own — with eight listeners it is noise, with eight hundred it is a result. So a MOS is only reportable with an interval attached.

Pure Python with `math` (`statistics` is allowed), no numpy.

`mos(ratings: list[float]) -> tuple[float, float, float]` returns `(mean, lo, hi)`, the 95% normal-approximation confidence interval for the mean:

```
mean   = (Σ r_i) / n
s      = sqrt( Σ (r_i − mean)² / (n − 1) )        sample standard deviation, n − 1 in the denominator
margin = 1.96 · s / sqrt(n)
lo, hi = mean − margin, mean + margin
```

- Use `n − 1`, not `n`. Dividing by `n` gives the standard deviation of *these* listeners; the interval is a claim about the listeners you did not ask, and the `n − 1` form is the unbiased estimate of that spread.
- Use the constant `1.96` exactly — a normal approximation, not a t-distribution quantile. (With 6 listeners the right multiplier is really 2.57; say so in the explain-back rather than coding it.)
- The interval is symmetric: `hi − mean == mean − lo` up to float error.
- If every rating is identical the margin is 0 and `lo == hi == mean`. That is an honest answer to a dishonest sample, not a bug.
- Raise `ValueError` if there are fewer than 2 ratings (with one listener there is no spread to estimate) or if any rating is outside `[1, 5]`.

## Examples

```
mos([1, 2, 3, 4, 5])          → (3.0, 1.6140707..., 4.3859293...)
      s = sqrt(2.5) = 1.5811388, margin = 1.96 · 1.5811388 / sqrt(5) = 1.3859293
      (dividing by n instead of n − 1 would give the margin 1.2396128 — a visibly narrower claim)

mos([4, 4, 5, 3, 4, 4])       → (4.0, 3.4939302..., 4.5060698...)
mos([1, 2, 3, 4, 5] * 4)      → (3.0, 2.3640920..., 3.6359080...)      same mean, interval less than half as wide
mos([4.0, 4.0, 4.0])          → (4.0, 4.0, 4.0)
mos([4.0])                    → ValueError
mos([4.0, 6.0])               → ValueError
```

## Constraints

- Up to 10 000 ratings. Ratings are floats in `[1, 5]`; they need not be whole numbers.
- Absolute tolerance 1e-9.

## Hints

1. Write the three numbers you need in order: the mean, the spread, and how the spread shrinks when you ask more listeners. Which one of the three has `n` in it twice?
2. Try `[1, 2, 3, 4, 5]` with `n` in the denominator of the variance and then with `n − 1`. Which gives the wider interval, and why is the wider one the honest one?
3. What goes wrong arithmetically with one rating, and what goes wrong in meaning?
4. If you quadruple the number of listeners and the ratings keep the same spread, by what factor does the margin shrink? Where in the formula does that factor come from?

## Explain-back

- Two systems score MOS 4.1 ± 0.3 and 4.3 ± 0.4. Can you say the second is better? What would you have to change about the listening test to be able to?
- Why is `1.96` the wrong multiplier for 6 listeners, and which direction is the error — does it make your interval too wide or too narrow?
- MOS is one of three standard TTS metrics. What do the other two (WER of an ASR system on the generated audio, and speaker similarity) measure that MOS does not, and when would a system score well on all three and still be unusable?
- Listeners who rate 40 utterances in a row drift, and listeners differ in how they use the scale. Which of those does this interval account for, and which does it quietly ignore?
