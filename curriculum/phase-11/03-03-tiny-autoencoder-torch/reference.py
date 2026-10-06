# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import torch
import torch.nn as nn
import torch.nn.functional as F


class AE(nn.Module):
    def __init__(self, d_in: int, d_latent: int) -> None:
        super().__init__()
        if d_in < 1 or d_latent < 1:
            raise ValueError("d_in and d_latent must be at least 1")
        self.encoder = nn.Linear(d_in, d_latent)
        self.decoder = nn.Linear(d_latent, d_in)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def train_ae(
    data: torch.Tensor, d_latent: int, steps: int = 600, lr: float = 0.05, seed: int = 0
) -> tuple[AE, float]:
    if data.dim() != 2:
        raise ValueError("data must be 2-D (N, d_in)")
    if steps < 1:
        raise ValueError("steps must be at least 1")

    torch.manual_seed(seed)
    model = AE(data.shape[1], d_latent)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.mse_loss(model(data), data)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return model, float(F.mse_loss(model(data), data))
