import torch


def kl_penalized_reward(r: torch.Tensor, logp_policy: torch.Tensor, logp_ref: torch.Tensor, beta: float) -> torch.Tensor:
    """Shape (B,): r minus beta times the per-sequence summed log-ratio policy/reference."""
    raise NotImplementedError


def kl_estimate(logp_policy: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """0-d tensor: the per-sequence summed log-ratio, averaged over the batch."""
    raise NotImplementedError
