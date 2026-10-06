# Where img2img starts

Topic: 6. Latent diffusion and text-to-image conditioning
Difficulty: 2 of 3

## Problem

Text-to-image starts the reverse process from pure Gaussian noise. img2img does not: it encodes your input image to a latent, noises that latent **part way** up the forward schedule, and then runs only the remaining denoising steps. The `strength` slider decides how far up it goes. Write the two functions that produce the starting point, in PyTorch (CPU, float32).

- `img2img_start_step(strength: float, T: int) -> int` — how many denoising steps are actually run:

  ```
  n_steps = floor(strength * T)        clamped into [0, T]
  ```

  `strength = 1.0` gives `T`, i.e. the whole schedule, which is exactly text-to-image. `strength = 0.0` gives `0`: no denoising runs and the output is the input image. Use `floor`, not rounding: `img2img_start_step(0.75, 50)` is `37`, not `38`. Raise `ValueError` if `strength` is outside `[0, 1]` or if `T < 1`.

- `noise_input_latent(latent: torch.Tensor, step: int, alpha_bars: torch.Tensor, eps: torch.Tensor) -> torch.Tensor` — the forward process in closed form, applied once to jump straight to timestep `step`:

  ```
  x = sqrt(alpha_bars[step]) * latent + sqrt(1 - alpha_bars[step]) * eps
  ```

  `alpha_bars` is a 1-D tensor of length `T` with `alpha_bars[0]` near 1 (almost no noise) decreasing to `alpha_bars[T-1]` near 0 (almost all noise). `eps` is standard Gaussian noise with the same shape as `latent`; the caller supplies it, so the function is deterministic. Return a new tensor of the same shape and dtype; do not modify the inputs.

  Raise `ValueError` if `step` is not in `[0, T-1]`, or if `eps` and `latent` have different shapes. Having run `n = img2img_start_step(strength, T)`, the caller noises to index `n - 1` and denoises down from there; when `n == 0` nothing is noised and `noise_input_latent` is never called.

Note what the formula does **not** say: the input latent never disappears unless `alpha_bars[step]` is 0. At `strength = 0.6` most of the input image's structure is still inside `x`, which is why img2img keeps the composition of the original. Banned: `diffusers` schedulers or any `add_noise` helper — write the two terms yourself.

## Examples

```
img2img_start_step(1.0, 50)   -> 50      full schedule = plain text-to-image
img2img_start_step(0.0, 50)   -> 0       output is the input, untouched
img2img_start_step(0.5, 50)   -> 25
img2img_start_step(0.75, 50)  -> 37      floor(37.5), not 38
img2img_start_step(1.5, 50)   -> ValueError

latent = [[3.0, 4.0]], eps = [[1.0, -1.0]], alpha_bars = [0.99, 0.64, 0.01]

noise_input_latent(latent, 0, alpha_bars, eps)
    -> sqrt(0.99)*[[3,4]] + sqrt(0.01)*[[1,-1]]  = [[3.085, 3.880]]   barely touched
noise_input_latent(latent, 1, alpha_bars, eps)
    -> 0.8*[[3,4]] + 0.6*[[1,-1]]                = [[3.000, 2.600]]
noise_input_latent(latent, 2, alpha_bars, eps)
    -> 0.1*[[3,4]] + sqrt(0.99)*[[1,-1]]         = [[1.295, -0.595]]  nearly pure noise
```

## Constraints

- Latent-shaped tensors, e.g. `(1, 4, 8, 8)`; any shape must work.
- float32 on CPU, tolerance 1e-5. `T` up to 1000.
- One vectorised expression per function; no Python loops over elements.

## Hints

1. At `strength = 1` the function must return `T`, and at `strength = 0` it must return `0`. Which of `floor`, `round` and `ceil` satisfy both, and which one does `0.75 * 50` separate?
2. The two coefficients in the forward process satisfy `a^2 + b^2 = 1`. Which one multiplies the latent, and what does it equal when `alpha_bars[step]` is near 1?
3. What would the output be if you dropped the `sqrt(alpha_bars[step])` factor and just added noise to the latent? Compare its variance with the formula's.
4. If img2img started from pure Gaussian noise, which term would have to vanish, and what would `strength` then control?

## Explain-back

- Someone implements img2img by sampling `x = randn_like(latent)` and running the last 60% of the steps with the prompt. What do they get out, and which term of the forward formula did they throw away?
- `strength = 1.0` and `strength = 0.0` are both degenerate. Say precisely what each produces and why neither is useful as an edit.
- The forward process can jump to any timestep in one line, but the reverse process cannot jump back in one line. What is different about the two directions?
- Inpainting keeps the unmasked region of the original latent at every step instead of only at the start. Why does the single noising step in this exercise not give you inpainting?
