import numpy as np


def linear_beta_schedule(T: int, b0: float, b1: float) -> np.ndarray:
    """T evenly spaced noise variances from b0 to b1, both endpoints included."""
    raise NotImplementedError


def alpha_bars(betas: np.ndarray) -> np.ndarray:
    """Running product of (1 - beta), so entry t is how much of x_0 survives after t + 1 steps."""
    raise NotImplementedError
