# Mode coverage

Topic: 4. GANs, briefly
Difficulty: 2 of 3

## Problem

A GAN can produce beautiful samples and still be broken: if the generator finds one output the discriminator cannot refute, it will happily emit variations of that one thing forever. Sample quality says nothing about sample diversity, so you measure them separately. On synthetic data whose true modes you know, the diversity measure is mode coverage.

`mode_coverage(samples: np.ndarray, centers: np.ndarray, radius: float) -> float`

- `samples` is a 2-D `float64` array of shape `(N, D)`: generated samples.
- `centers` is a 2-D `float64` array of shape `(M, D)`: the centres of the `M` true modes.
- `radius` is a non-negative `float`.

A mode is **covered** when at least one sample lies within Euclidean distance `radius` of its centre, with the boundary counting as covered (`distance <= radius`). Return the fraction of modes covered: `covered / M`, a `float` in `[0, 1]`.

Read that denominator carefully. It is the number of **modes**, not the number of samples. Ten thousand samples piled on one of eight modes scores `0.125`, not `1.0` — a measure of how much of the data distribution the generator reaches, deliberately blind to how many samples it spends getting there. A single sample is enough to cover a mode, and extra samples on an already-covered mode add nothing.

Do not modify the inputs. Raise `ValueError` if either array is not 2-D, if their widths disagree, if either has zero rows, or if `radius < 0`.

## Examples

```
centers = array([[0., 0.], [10., 0.], [0., 10.], [10., 10.]])

mode_coverage(centers, centers, 0.0)                       → 1.0     one sample per mode is plenty
mode_coverage(array([[0., 0.]] * 500), centers, 1.0)       → 0.25    textbook mode collapse
mode_coverage(vstack([[[0., 0.]] * 50, [[10., 0.]]]), centers, 1.0)
                                                           → 0.5     51 samples, 2 modes
mode_coverage(array([[3., 4.]]), array([[0., 0.]]), 5.0)   → 1.0     distance exactly 5, inclusive
mode_coverage(array([[3., 4.]]), array([[0., 0.]]), 4.999) → 0.0
mode_coverage(array([[50., 50.]]), centers, 1.0)           → 0.0     far from everything

mode_coverage(centers, centers, -1.0)                      → ValueError
```

## Constraints

- `N` up to 10000, `M` up to 64, `D` up to 16.
- numpy only; broadcast rather than looping over `N * M` pairs in Python.
- Distances compared within `1e-9`; the `distance <= radius` boundary is exact in the tests.

## Hints

1. Which quantity do you need per (sample, mode) pair, and what shape is the table of them?
2. Once you have that table, which axis do you reduce over to answer "was this mode reached by anybody", and which reduction — `any`, `all`, `sum`, `mean`?
3. The answer divides by `M`. If you divided by `N` instead, what would 10000 samples sitting on one of eight modes score, and which of the two numbers is the one that detects collapse?
4. Why does comparing squared distance against `radius ** 2` give the same answer, and what does it do to the exact boundary case?

## Explain-back

- Mode coverage says nothing about how good the samples look, and FID would not clearly separate these cases either. What does each one see that the other misses?
- Why does the adversarial game have mode collapse as a stable outcome at all? What is the generator being rewarded for?
- Picking `radius` is a judgement call. What does a radius far too large report, and what does one far too small report?
- You cannot write this function for real photographs, because nobody handed you the centres. What stands in for it there, and what does that substitution cost you?
