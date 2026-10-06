# Beta schedule and alpha bars

Topic: 5. Diffusion models: DDPM to DDIM
Difficulty: 1 of 3

## Problem

Diffusion has three sets of numbers that people constantly confuse: `beta_t`, the variance of the noise added at step `t`; `alpha_t = 1 - beta_t`, what survives one step; and `alpha_bar_t`, the product of all the `alpha`s up to `t`, which is what survives from the original image. Build the first and the third.

### `linear_beta_schedule(T: int, b0: float, b1: float) -> np.ndarray`

`T` evenly spaced values from `b0` to `b1`, **both endpoints included**: `betas[0] == b0`, `betas[T - 1] == b1`, and the gap between consecutive entries is constant. Shape `(T,)`, `float64`. DDPM's defaults are `T = 1000`, `b0 = 1e-4`, `b1 = 0.02`.

Raise `ValueError` if `T < 2`, if `b0 <= 0`, if `b1 >= 1`, or if `b0 > b1`.

### `alpha_bars(betas: np.ndarray) -> np.ndarray`

```
alpha_t     = 1 - beta_t
alpha_bar_t = alpha_0 * alpha_1 * ... * alpha_t        (a running PRODUCT, not a sum)
```

Shape `(T,)`, same length as `betas`. Index `t` here means "after `t + 1` noising steps", so this array is 0-based and lines up with `betas`:

- `alpha_bars(betas)[0] == 1 - betas[0]`, **not** `1.0`. One step of noise has already been applied by index 0.
- `alpha_bars(betas)[t] == alpha_bars(betas)[t - 1] * (1 - betas[t])` for every later `t`.

Every entry lies in `(0, 1)` and the array is strictly decreasing, because each `alpha` is strictly between 0 and 1. With DDPM's defaults it starts at `0.9999` and ends around `4e-5`, which is why `x_T` is indistinguishable from pure noise: the coefficient `sqrt(alpha_bar_T)` on the original image is about `0.0064`.

Raise `ValueError` if `betas` is not 1-D, is empty, or holds any value outside `(0, 1)`. Do not modify the input.

## Examples

```
linear_beta_schedule(5, 0.1, 0.5)      → [0.1, 0.2, 0.3, 0.4, 0.5]
linear_beta_schedule(2, 0.1, 0.5)      → [0.1, 0.5]
linear_beta_schedule(1000, 1e-4, 0.02)[0], [-1]   → 1e-4, 0.02
linear_beta_schedule(5, 0.5, 0.1)      → ValueError
linear_beta_schedule(5, 0.1, 1.0)      → ValueError

alpha_bars(array([0.1, 0.2]))          → [0.9, 0.72]           0.9, then 0.9 * 0.8
                                                               (a cumulative sum would say 0.9, 1.7)
alpha_bars(array([0.5, 0.5, 0.5]))     → [0.5, 0.25, 0.125]
alpha_bars(linear_beta_schedule(1000, 1e-4, 0.02))[-1]  → 4.0358e-05
alpha_bars(array([0.0, 0.5]))          → ValueError
```

## Constraints

- `T` up to 4000.
- numpy only.
- Compared within `1e-12` on small schedules and `1e-9` relative on the 1000-step one.

## Hints

1. "Both endpoints included" rules out one of the two obvious numpy calls for evenly spaced values. Which one, and what is the spacing in terms of `T`?
2. What is `alpha_t` in terms of `beta_t`, and which single numpy function turns an array into its running product?
3. The array is 0-based. After how many noising steps are you at index 0, and what does that make `alpha_bars[0]`?
4. `alpha_bar` falls from about 1 to about 0 over the schedule. Which direction does `beta` move, and why does multiplying a thousand numbers each slightly below 1 end up so close to zero?

## Explain-back

- State in one sentence each what `beta_t`, `alpha_t` and `alpha_bar_t` mean. Which of the three appears in the closed-form jump from `x_0` to `x_t`?
- Why is `alpha_bar` a product rather than a sum? What would the forward process look like if it were a sum?
- `alpha_bar_T` is about `4e-5`, not exactly 0. What does that leftover mean for `x_T`, and why does sampling still start from pure noise?
- The cosine schedule was introduced because the linear one destroys information too early. Looking at the `alpha_bar` curve, where would you want to spend more of the schedule and why?
