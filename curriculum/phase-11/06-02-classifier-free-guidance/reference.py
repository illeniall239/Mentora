# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import torch


def cfg(eps_uncond: torch.Tensor, eps_cond: torch.Tensor, scale: float) -> torch.Tensor:
    if eps_uncond.shape != eps_cond.shape:
        raise ValueError("eps_uncond and eps_cond must have the same shape")
    return eps_uncond + scale * (eps_cond - eps_uncond)


def cfg_negative(eps_neg: torch.Tensor, eps_cond: torch.Tensor, scale: float) -> torch.Tensor:
    return cfg(eps_neg, eps_cond, scale)
