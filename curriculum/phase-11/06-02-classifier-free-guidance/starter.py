import torch


def cfg(eps_uncond: torch.Tensor, eps_cond: torch.Tensor, scale: float) -> torch.Tensor:
    """Classifier-free guidance: eps_uncond + scale * (eps_cond - eps_uncond)."""
    raise NotImplementedError


def cfg_negative(eps_neg: torch.Tensor, eps_cond: torch.Tensor, scale: float) -> torch.Tensor:
    """Same combination with the negative-prompt prediction in the unconditional slot."""
    raise NotImplementedError
