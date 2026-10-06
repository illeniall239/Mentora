# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import torch


def reparameterize(mu: torch.Tensor, log_var: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    if mu.shape != log_var.shape or mu.shape != eps.shape:
        raise ValueError("mu, log_var and eps must have the same shape")
    std = torch.exp(0.5 * log_var)
    return mu + std * eps


def kl_gaussian(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    if mu.shape != log_var.shape:
        raise ValueError("mu and log_var must have the same shape")
    if mu.dim() != 2:
        raise ValueError("mu and log_var must be 2-D (B, D)")
    return -0.5 * torch.sum(1.0 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
