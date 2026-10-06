# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import torch
import torch.nn.functional as F


def dpo_loss(pi_c: torch.Tensor, pi_r: torch.Tensor, ref_c: torch.Tensor, ref_r: torch.Tensor, beta: float) -> torch.Tensor:
    tensors = (pi_c, pi_r, ref_c, ref_r)
    if any(t.ndim != 1 for t in tensors) or any(t.shape != pi_c.shape for t in tensors):
        raise ValueError("pi_c, pi_r, ref_c and ref_r must all be 1-D tensors of the same shape")
    if pi_c.numel() == 0:
        raise ValueError("the batch must not be empty")
    if beta <= 0:
        raise ValueError("beta must be positive")
    chosen_ratio = pi_c - ref_c  # the implicit reward of each response, up to beta
    rejected_ratio = pi_r - ref_r
    logits = beta * (chosen_ratio - rejected_ratio)  # beta scales inside the log sigma
    return -F.logsigmoid(logits).mean()
