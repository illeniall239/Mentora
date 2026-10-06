# Classifier-free guidance

Topic: 6. Latent diffusion and text-to-image conditioning
Difficulty: 2 of 3

## Problem

At every denoising step Stable Diffusion runs its UNet **twice**: once with the prompt embedding and once with an empty one. It then combines the two noise predictions, pushing the step away from what the model would have done with no prompt at all. That is classifier-free guidance, and it is why a `guidance_scale` slider exists. Write the combination in PyTorch (CPU, float32).

- `cfg(eps_uncond: torch.Tensor, eps_cond: torch.Tensor, scale: float) -> torch.Tensor`

  ```
  eps = eps_uncond + scale * (eps_cond - eps_uncond)
  ```

  So `scale = 1` returns exactly `eps_cond` (guidance off — just normal conditional sampling), `scale = 0` returns exactly `eps_uncond` (the prompt is ignored), and `scale > 1` **extrapolates past** `eps_cond` along the line from `eps_uncond` to `eps_cond`. It is an extrapolation, not a blend: the result is not restricted to lie between the two inputs.

- `cfg_negative(eps_neg: torch.Tensor, eps_cond: torch.Tensor, scale: float) -> torch.Tensor` — the negative-prompt variant. A negative prompt is **not** a third term added on top: the pipeline encodes it instead of the empty string and feeds the result into the unconditional slot, so the same formula applies with `eps_neg` in place of `eps_uncond`. The step is therefore pushed directly away from the negative prompt's prediction.

Both return a **new** tensor with the same shape and dtype as the inputs, and must not modify the inputs in place. Both raise `ValueError` if the two tensors do not have the same shape. `scale` may be any finite float, including negative. Allowed: ordinary tensor arithmetic or `torch.lerp`. Banned: any `diffusers` import or guidance helper.

## Examples

```
eps_uncond = [[1.0, 0.0]], eps_cond = [[3.0, 4.0]]

cfg(eps_uncond, eps_cond, 0.0)   -> [[1.0, 0.0]]        the prompt does nothing
cfg(eps_uncond, eps_cond, 1.0)   -> [[3.0, 4.0]]        exactly the conditional prediction
cfg(eps_uncond, eps_cond, 2.0)   -> [[5.0, 8.0]]        past the conditional, not between
cfg(eps_uncond, eps_cond, 7.5)   -> [[16.0, 30.0]]      a typical Stable Diffusion setting

eps_neg = [[-2.0, 0.0]], eps_cond = [[3.0, 4.0]]
cfg_negative(eps_neg, eps_cond, 1.0)  -> [[3.0, 4.0]]
cfg_negative(eps_neg, eps_cond, 3.0)  -> [[13.0, 12.0]]
```

## Constraints

- Latent-shaped tensors, e.g. `(1, 4, 8, 8)`; any shape must work, including 1-D.
- float32 on CPU, tolerance 1e-6.
- No loops over elements: this is one vectorised expression.

## Hints

1. Write the formula with `scale = 1` substituted in and simplify. What is left? Now do the same with `scale = 0`.
2. `eps_cond - eps_uncond` is a direction in noise space. What does it point towards, and what does multiplying it by 7.5 do to the length of the step taken from `eps_uncond`?
3. A blend `(1 - s) * a + s * b` with `s` in [0, 1] always lands between `a` and `b`. Is your formula the same expression? What happens to it when `s = 7.5`?
4. If a negative prompt needed its own extra term, the pipeline would have to run the UNet three times per step. It runs it twice. Which of the two predictions does the negative prompt supply?

## Explain-back

- Guidance scale 1 means "no guidance", not "no prompt". Why is the useless setting 1 rather than 0, and what does scale 0 actually produce?
- Users often report that scale 15 gives burnt, oversaturated, strangely similar images. What does extrapolating far past `eps_cond` do to the sample, and what is the trade-off against prompt adherence?
- CFG doubles the cost of every denoising step. What exactly is computed twice, and what does a model distilled for guidance (so it needs only one pass) save?
- A user adds "blurry, watermark" as a negative prompt. Trace what changes inside the step, and explain why that is different from deleting those words from the positive prompt.
