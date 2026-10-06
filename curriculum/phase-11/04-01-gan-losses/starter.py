import numpy as np


def d_loss(d_real: np.ndarray, d_fake: np.ndarray) -> float:
    """Discriminator loss: -mean(log D(x)) - mean(log(1 - D(G(z))))."""
    raise NotImplementedError


def g_loss_nonsat(d_fake: np.ndarray) -> float:
    """Non-saturating generator loss: -mean(log D(G(z)))."""
    raise NotImplementedError
