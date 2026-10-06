# L_simple, x0 prediction and one DDIM step

Topic: 5. Diffusion models: DDPM to DDIM
Difficulty: 3 of 3

## Problem

A diffusion model's network predicts the noise. That one choice gives you a trivial training loss, a one-line route back to a clean-image estimate, and a deterministic sampler that can skip most of the schedule. Write all three in numpy.

### `ddpm_loss(eps: np.ndarray, eps_pred: np.ndarray) -> float`

```
L_simple = mean over all elements of (eps - eps_pred) ** 2
```

Plain mean squared error between the noise that was actually added and the noise the network predicted — no weighting by `t`, no KL terms, no `log` anywhere; that is what "simple" means. **Mean**, not sum, so the number does not depend on the size of the tensor. It is `0.0` when the prediction is exact, and symmetric in its two arguments. Return a Python `float`.

### `predict_x0(x_t: np.ndarray, eps_pred: np.ndarray, abar: float) -> np.ndarray`

Invert the closed-form forward step `x_t = sqrt(abar) * x_0 + sqrt(1 - abar) * eps`, solving for `x_0`:

```
x0_hat = (x_t - sqrt(1 - abar) * eps_pred) / sqrt(abar)
```

Same shape as `x_t`. With the true `eps` this returns `x_0` exactly — the model never predicts the image directly, it predicts the noise and the image falls out algebraically. Note how unstable it is for small `abar`: dividing by `sqrt(abar) = 0.0064` at the end of the schedule amplifies any error in `eps_pred` by about 157, which is why a single step from pure noise does not produce an image.

### `ddim_step(x_t: np.ndarray, eps_pred: np.ndarray, abar_t: float, abar_prev: float) -> np.ndarray`

One deterministic DDIM update from noise level `abar_t` to any earlier (larger) `abar_prev`:

```
ddim_step = sqrt(abar_prev) * predict_x0(x_t, eps_pred, abar_t) + sqrt(1 - abar_prev) * eps_pred
```

Estimate the clean image, then re-noise it to the target level using the **same** `eps_pred`. There is no random term: DDIM is deterministic, so calling it twice on the same inputs gives bitwise identical results, and the same starting noise always gives the same image. That is also why `abar_prev` is free — it does not have to be the neighbouring step, which is how 1000 training steps become 20 or 50 sampling steps.

Three identities the tests rely on, all exact when `eps_pred` is the true `eps`:

- `abar_prev == abar_t` returns `x_t` unchanged.
- `abar_prev == 1.0` returns `predict_x0(...)`, i.e. `x_0`.
- Stepping from `x_t` to any `abar_prev` lands exactly on `sqrt(abar_prev) * x_0 + sqrt(1 - abar_prev) * eps` — the same point the forward process would have produced at that level with the same noise. Chain those steps down a subsequence of the schedule, ending at `abar_prev = 1.0`, and you recover `x_0` to floating-point precision.

Do not modify the inputs. Raise `ValueError` if the two array arguments have different shapes, or if `abar`, `abar_t` or `abar_prev` is outside `(0, 1]`.

## Examples

```
ddpm_loss(array([1., 1.]), array([1., 1.]))            → 0.0
ddpm_loss(array([0., 0.]), array([1., -1.]))           → 1.0            mean, not sum
ddpm_loss(zeros(100), ones(100))                       → 1.0            and not 100.0

predict_x0(array([1.4]), array([1.]), 0.64)            → [1.0]          0.8 * 1 + 0.6 * 1 = 1.4
predict_x0(array([0.6]), array([1.]), 0.64)            → [0.0]

ddim_step(array([1.4]), array([1.]), 0.64, 0.64)       → [1.4]          same level, no move
ddim_step(array([1.4]), array([1.]), 0.64, 1.0)        → [1.0]          all the way to x0
ddim_step(array([1.4]), array([1.]), 0.64, 0.25)       → [1.3660254]    == 0.5 * 1 + sqrt(0.75) * 1

predict_x0(x_t, eps_pred, 0.0)                         → ValueError
ddim_step(x_t, eps_pred, 0.5, 1.5)                     → ValueError
```

## Constraints

- Arrays of any shape, up to about 100000 elements.
- numpy only. No randomness anywhere in these three functions.
- Compared within `1e-10`.

## Hints

1. `x_t = sqrt(abar) * x0 + sqrt(1 - abar) * eps` is one equation in one unknown once `eps_pred` stands in for `eps`. Which operations undo it, and in which order?
2. What does `predict_x0` do to an error in `eps_pred` when `abar` is `4e-5`? Write down the factor, and then say why sampling takes many steps.
3. Read the DDIM formula as two moves: which part of it is "where do I think the clean image is" and which part is "put the noise back"? What noise does it put back — a fresh draw, or the one already in hand?
4. Set `abar_prev = abar_t` and simplify the expression by hand. What has to come out, and what does that tell you about where the `sqrt(1 - abar_prev)` term comes from?

## Explain-back

- The network predicts `eps`, yet what you want is an image. Trace how `x_0` is obtained, and say what would change if the network predicted `x_0` directly instead.
- Why does sampling need many steps if `predict_x0` can already produce an image estimate from `x_T` in one line? Be specific about what is wrong with that estimate.
- DDIM is deterministic and DDPM's reverse step adds noise. What do you gain from dropping that noise, and what do you lose?
- "More sampling steps are always better." Using `ddim_step`, say what more steps actually buy and where the returns stop.
