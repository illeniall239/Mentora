import torch


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average hidden (B, T, D) over the positions where mask (B, T) is on, then L2-normalize; (B, D)."""
    raise NotImplementedError
