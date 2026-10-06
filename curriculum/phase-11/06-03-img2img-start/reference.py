# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import math

import torch


def img2img_start_step(strength: float, T: int) -> int:
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if T < 1:
        raise ValueError("T must be at least 1")
    return min(int(math.floor(strength * T)), T)


def noise_input_latent(
    latent: torch.Tensor, step: int, alpha_bars: torch.Tensor, eps: torch.Tensor
) -> torch.Tensor:
    if not 0 <= step < alpha_bars.numel():
        raise ValueError("step must be in [0, len(alpha_bars) - 1]")
    if latent.shape != eps.shape:
        raise ValueError("latent and eps must have the same shape")
    abar = alpha_bars[step]
    return abar.sqrt() * latent + (1.0 - abar).sqrt() * eps
