import torch
import torch.nn as nn


class AE(nn.Module):
    """A linear encoder/decoder pair with a d_latent bottleneck."""

    def __init__(self, d_in: int, d_latent: int) -> None:
        raise NotImplementedError

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map (N, d_in) inputs to (N, d_latent) codes."""
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map (N, d_latent) codes back to (N, d_in) reconstructions."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """decode(encode(x))."""
        raise NotImplementedError


def train_ae(
    data: torch.Tensor, d_latent: int, steps: int = 600, lr: float = 0.05, seed: int = 0
) -> tuple["AE", float]:
    """Seed, build an AE, run full-batch Adam on the MSE, and return the model and its final loss."""
    raise NotImplementedError
