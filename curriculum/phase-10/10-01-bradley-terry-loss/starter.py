import torch


def bradley_terry_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    """Mean of -log sigma(r_chosen - r_rejected) over the batch, as a 0-d tensor."""
    raise NotImplementedError
