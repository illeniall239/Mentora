# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import torch
import torch.nn.functional as F


def bradley_terry_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    if r_chosen.ndim != 1 or r_rejected.ndim != 1:
        raise ValueError("rewards must be 1-D tensors of shape (n,)")
    if r_chosen.shape != r_rejected.shape:
        raise ValueError("chosen and rejected rewards must have the same shape")
    if r_chosen.numel() == 0:
        raise ValueError("the batch must not be empty")
    margin = r_chosen - r_rejected  # only the difference matters
    return -F.logsigmoid(margin).mean()  # logsigmoid, so a margin of -200 stays finite
