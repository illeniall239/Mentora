# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import torch


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape != hidden.shape[:2]:
        raise ValueError(f"mask {tuple(mask.shape)} does not match hidden {tuple(hidden.shape[:2])}")
    m = mask.to(hidden.dtype)
    counts = m.sum(dim=1, keepdim=True)
    if (counts == 0).any():
        raise ValueError("a sequence has no real token to pool")
    summed = (hidden * m.unsqueeze(-1)).sum(dim=1)
    mean = summed / counts
    return mean / mean.norm(dim=1, keepdim=True)
