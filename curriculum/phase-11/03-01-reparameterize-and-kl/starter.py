import torch


def reparameterize(mu: torch.Tensor, log_var: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """Sample z = mu + sigma * eps with sigma = exp(0.5 * log_var), keeping gradients to mu and log_var."""
    raise NotImplementedError


def kl_gaussian(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Closed-form KL(N(mu, sigma^2) || N(0, I)) per example, summed over the last dimension."""
    raise NotImplementedError
