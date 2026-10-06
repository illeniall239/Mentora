import torch


def img2img_start_step(strength: float, T: int) -> int:
    """Number of denoising steps actually run: floor(strength * T), clamped to [0, T]."""
    raise NotImplementedError


def noise_input_latent(
    latent: torch.Tensor, step: int, alpha_bars: torch.Tensor, eps: torch.Tensor
) -> torch.Tensor:
    """Forward process in closed form: sqrt(abar)*latent + sqrt(1 - abar)*eps at index step."""
    raise NotImplementedError
