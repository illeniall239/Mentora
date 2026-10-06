# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import torch


def _sequence_kl(logp_policy: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    if logp_policy.ndim != 2 or logp_ref.ndim != 2:
        raise ValueError("log-probs must be 2-D tensors of shape (B, T)")
    if logp_policy.shape != logp_ref.shape:
        raise ValueError("policy and reference log-probs must have the same shape")
    if logp_policy.numel() == 0:
        raise ValueError("the batch must not be empty")
    return (logp_policy - logp_ref).sum(dim=1)  # one KL estimate per sequence; may be negative


def kl_penalized_reward(r: torch.Tensor, logp_policy: torch.Tensor, logp_ref: torch.Tensor, beta: float) -> torch.Tensor:
    kl = _sequence_kl(logp_policy, logp_ref)
    if r.ndim != 1 or r.shape != kl.shape:
        raise ValueError("r must be 1-D of shape (B,)")
    if beta < 0:
        raise ValueError("beta must be non-negative")
    return r - beta * kl  # drifting from the reference costs reward


def kl_estimate(logp_policy: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    return _sequence_kl(logp_policy, logp_ref).mean()
