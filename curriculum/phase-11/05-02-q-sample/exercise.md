# q_sample: jumping to any noise level in one shot

Topic: 5. Diffusion models: DDPM to DDIM
Difficulty: 2 of 3

## Problem

The forward process adds a little noise at a time: `q(x_t | x_{t-1}) = N(sqrt(1 - beta_t) * x_{t-1}, beta_t * I)`. Running it step by step to reach `t = 700` would be hopeless for training. The whole trick is that the composition of all those Gaussians is itself a Gaussian with a known mean and variance, so you can jump straight there:

```
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,      eps ~ N(0, I)
```

`q_sample(x0: np.ndarray, t: int, eps: np.ndarray, alpha_bars: np.ndarray) -> np.ndarray`

- `x0` is a `float64` array of any shape — a pixel image, a batch, a `(4, 64, 64)` latent, whatever.
- `t` is a Python `int`, a 0-based index into `alpha_bars`. Index `t` is the same index as in the schedule you built from the betas, so `t = 0` is one step of noise and `t = len(alpha_bars) - 1` is the last.
- `eps` is the noise, the same shape as `x0`, supplied by the caller so these tests are deterministic. It must be a draw from the **standard** normal `N(0, I)`: all the scaling is in the two coefficients.
- `alpha_bars` is the 1-D cumulative-product schedule, strictly inside `(0, 1]`.

Return a new array with the same shape as `x0`. Note which quantities are under the square roots: `sqrt(alpha_bar_t)` scales the image and `sqrt(1 - alpha_bar_t)` scales the noise — not `alpha_bar_t` and `1 - alpha_bar_t` themselves. That choice is what makes the process **variance preserving**: since the two coefficients obey `a ** 2 + b ** 2 = 1`, if `x0` has unit variance then so does `x_t`, at every `t`. Get the square roots wrong and the signal quietly changes scale as `t` moves, which the network then has to absorb.

Two consequences the tests check:

- With DDPM's schedule, `t = 0` gives `0.99995 * x0 + 0.01 * eps`, visually `x0`.
- At the last `t`, `sqrt(alpha_bar_t)` is about `0.0064`, so `x_t` is essentially `eps`: mean 0, variance 1, almost nothing of `x0` left.

Do not modify the inputs. Raise `ValueError` if `eps` and `x0` have different shapes, if `alpha_bars` is not 1-D or is empty or holds a value outside `(0, 1]`, or if `t` is not an integer in `[0, len(alpha_bars))`.

## Examples

```
abars = array([0.64, 0.25])

q_sample(array([1., 1.]), 0, array([0., 0.]), abars)   → [0.8, 0.8]        sqrt(0.64) = 0.8
q_sample(array([0., 0.]), 0, array([1., 1.]), abars)   → [0.6, 0.6]        sqrt(1 - 0.64) = 0.6
q_sample(array([1., 1.]), 0, array([1., 1.]), abars)   → [1.4, 1.4]
q_sample(array([1., 1.]), 1, array([1., 1.]), abars)   → [1.3660254, 1.3660254]

0.8 ** 2 + 0.6 ** 2                                    → 1.0               variance preserving

q_sample(x0, 2, eps, abars)                            → ValueError        t is out of range
q_sample(x0, -1, eps, abars)                           → ValueError
q_sample(x0, 0, eps_of_a_different_shape, abars)       → ValueError
```

## Constraints

- `x0` up to about 100000 elements; `len(alpha_bars)` up to 4000.
- numpy only. No sampling inside the function: `eps` comes in as an argument.
- Compared within `1e-12`, and statistical checks within `0.05` over 20000 seeded draws.

## Hints

1. Which of `alpha_bar_t` and `1 - alpha_bar_t` belongs to `x0` and which to `eps`? What happens to the output as `alpha_bar_t` approaches 0?
2. Set `eps` to all zeros and then `x0` to all zeros. What does each call isolate, and what does that give you to check against by hand?
3. If `x0` and `eps` are independent with unit variance, the variance of `a * x0 + b * eps` is `a ** 2 + b ** 2`. What do `a` and `b` have to be for that to come out as 1 at every `t`?
4. `t` indexes `alpha_bars`. Which value does `t = 0` pick out, and is that the clean image or the image after one noising step?

## Explain-back

- Why can the forward process jump from `x_0` straight to `x_700` while the reverse process cannot jump from `x_700` straight to `x_0`?
- What exactly is the training target of the network here, given `x_t` and `t`? Why is predicting that easier than predicting the clean image directly?
- The coefficients satisfy `a ** 2 + b ** 2 = 1`. What would break if you used `alpha_bar_t` and `1 - alpha_bar_t` without the square roots?
- `eps` must be standard normal. What would drawing it from `N(0, 4I)` do to `x_t`, and what would the network learn to predict instead?
