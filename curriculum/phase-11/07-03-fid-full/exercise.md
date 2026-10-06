# FID from feature matrices

Topic: 7. Evaluating image models
Difficulty: 3 of 3

## Problem

FID takes the Inception activations of two sets of images, fits a Gaussian to each, and reports the squared Fréchet distance between them. Here the features are handed to you as plain arrays — no Inception, no images — so what is left is the part people get wrong: the matrix square root. Write it in numpy.

`fid(feats_a: np.ndarray, feats_b: np.ndarray) -> float`, where `feats_a` has shape `(n, d)` and `feats_b` has shape `(m, d)`: `n` and `m` may differ, `d` must match.

```
mu_a, mu_b  = column means
C_a, C_b    = sample covariances, ddof = 1   (np.cov(x, rowvar=False, ddof=1))
fid = ||mu_a - mu_b||^2 + tr(C_a) + tr(C_b) - 2 * tr( (C_a C_b)^(1/2) )
```

The last term is the hard one, because `C_a C_b` is generally **not symmetric**, so you cannot eigendecompose it directly and trust the result. Use the standard identity

```
tr( (C_a C_b)^(1/2) ) = tr( (C_a^(1/2) C_b C_a^(1/2))^(1/2) )
```

where the inner matrix **is** symmetric positive semi-definite. So:

1. Eigendecompose the symmetric `C_a` with `np.linalg.eigh`, clamp its eigenvalues at 0, and rebuild `C_a^(1/2) = V diag(sqrt(w)) V^T`.
2. Form `M = C_a^(1/2) C_b C_a^(1/2)` and symmetrise it (`(M + M.T) / 2`) to undo rounding error.
3. Take its eigenvalues with `np.linalg.eigvalsh`, clamp them at 0, and sum their square roots. That sum is the trace term.

Return a Python `float`. Clamp the final result at 0 so rounding cannot return a tiny negative number.

Raise `ValueError` if the two feature dimensions differ, or if either set has fewer than 2 rows (a sample covariance with `ddof = 1` needs at least two samples).

Allowed: `np.cov`, `np.linalg.eigh`, `np.linalg.eigvalsh`, `np.trace`, ordinary array arithmetic. Banned: `scipy.linalg.sqrtm`, `torchmetrics`, `pytorch-fid`, or any library FID.

## Examples

```
a = [[1,0],[-1,0],[0,2],[0,-2]]              mean 0, C_a = diag(2/3, 8/3)

fid(a, a)       -> 0.0                       identical feature sets
fid(a, a + 3)   -> 18.0                      mean shift of (3,3): 3^2 + 3^2
fid(a, 3 * a)   -> 13.333...                 C_b = 9 C_a, so 4*(2/3) + 4*(8/3) = 40/3
fid(a, a[::-1]) -> 0.0                       row order carries no information

fid(a, b) with b of shape (5, 3)   -> ValueError    feature dimensions differ
fid(a, a[:1])                      -> ValueError    one sample has no covariance
```

## Constraints

- numpy only; `d` from 2 to 64, `n`, `m` from 2 to about 5000. Tolerance 1e-6.
- `fid(a, b)` equals `fid(b, a)` and is never negative.
- Must run in well under a second: an eigendecomposition of a `d x d` matrix, not of anything `n x n`.

## Hints

1. Why can `C_a C_b` have complex eigenvalues even though both factors are symmetric? What does `np.linalg.eig` give you there that you cannot sum square roots of?
2. `eigh` returns `(w, V)` for a symmetric matrix. How do you rebuild a function of that matrix — the square root, say — from `w` and `V` without a loop?
3. A covariance estimated from finitely many samples can have eigenvalues like `-3e-17`. What does `sqrt` do with those, and where exactly should you clamp?
4. You only ever need the *trace* of the inner square root, never the matrix itself. Which of the two eigen routines do you need for that step, and what work does that save?

## Explain-back

- The paper you are comparing against used 50 000 samples; you have 500. Which direction does your FID move, and why is it not just "noisier"?
- Two labs compute FID on the same model and get 9.1 and 11.4. List what could differ (feature extractor version, image resizing, JPEG vs PNG, sample count) and say which of those changes the *features* rather than the *statistics*.
- FID compares only a mean and a covariance in feature space. Construct a failure: a generator whose FID is excellent and whose samples you would still reject.
- Your implementation eigendecomposes a `d x d` matrix. What breaks when `d = 2048` and you only have 1000 samples, and what does that do to the rank of `C_a`?
