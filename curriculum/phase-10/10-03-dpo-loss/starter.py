import torch


def dpo_loss(pi_c: torch.Tensor, pi_r: torch.Tensor, ref_c: torch.Tensor, ref_r: torch.Tensor, beta: float) -> torch.Tensor:
    """Mean of -log sigma(beta * ((pi_c - ref_c) - (pi_r - ref_r))) over the batch, as a 0-d tensor."""
    raise NotImplementedError
