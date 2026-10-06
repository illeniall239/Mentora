# Fréchet distance in one dimension

Topic: 7. Evaluating image models
Difficulty: 1 of 3

## Problem

FID fits one Gaussian to the Inception features of the real images and another to the generated ones, then reports the squared Fréchet distance between those two Gaussians. Before writing the 2048-dimensional version, write the scalar one, where every matrix collapses to a number.

`frechet_distance_1d(mu1: float, s1: float, mu2: float, s2: float) -> float` returns

```
d2 = (mu1 - mu2) ** 2 + s1 + s2 - 2 * sqrt(s1 * s2)
```

`s1` and `s2` are **variances** (sigma squared), matching the covariance matrices in the full formula `||mu1 - mu2||^2 + Tr(C1 + C2 - 2 (C1 C2)^(1/2))`, not standard deviations. Return the squared distance `d2` — that is the number FID reports; do not take a square root at the end.

Rules:

- Variances may be `0` (a point mass); the formula still works.
- Raise `ValueError` if `s1 < 0` or `s2 < 0`: a negative variance is not a Gaussian.
- Plain Python with `math`. No numpy, no scipy, no `scipy.linalg.sqrtm`.

Note what the result is and is not: it is 0 exactly when the two Gaussians are identical, it is symmetric in the two arguments, it never goes negative, and it has **no upper bound and no units of its own** — it is measured in whatever the feature space is measured in. That is the first reason an FID of 12 in one paper and 12 in another are not comparable.

## Examples

```
frechet_distance_1d(0.0, 1.0, 0.0, 1.0)  -> 0.0     identical Gaussians
frechet_distance_1d(0.0, 1.0, 3.0, 1.0)  -> 9.0     means 3 apart, same spread
frechet_distance_1d(0.0, 1.0, 0.0, 4.0)  -> 1.0     same mean, sigma 1 vs sigma 2
frechet_distance_1d(0.0, 1.0, 3.0, 4.0)  -> 10.0    both terms add
frechet_distance_1d(0.0, 0.0, 2.0, 0.0)  -> 4.0     two point masses
frechet_distance_1d(0.0, 1.0, 0.0, -1.0) -> ValueError
```

## Constraints

- Plain Python floats; tolerance 1e-9.
- Means and variances up to about 1e6 in magnitude.

## Hints

1. `s1 + s2 - 2 * sqrt(s1 * s2)` is a perfect square in disguise. Of what? And what does that tell you about the sign of the result?
2. The two Gaussians have sigma 1 and sigma 2. Is the variance term 1, 3 or 9? Work it out both ways and see which one your formula gives.
3. What has to be true of `mu1`, `mu2`, `s1` and `s2` for the answer to be exactly 0? Is a matching mean enough?
4. Which argument of the full matrix formula does `s1` stand in for: a covariance matrix or its square root?

## Explain-back

- Two papers both report "FID 12" on the same dataset. Name three things that could differ between them and still leave both numbers honest.
- The distance uses only a mean and a variance. What property of a generated distribution can be badly wrong while this number stays small?
- Estimating a variance from 50 samples is noisy, and the bias does not average out. Which way does a small sample push FID, and why is "FID on 100 images" not a smaller-scale version of the real thing?
- In the full version `s1` and `s2` are matrices and the last term needs a matrix square root. Which part of your one-line formula becomes expensive, and which part stays trivial?
