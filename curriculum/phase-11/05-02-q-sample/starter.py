import numpy as np


def q_sample(x0: np.ndarray, t: int, eps: np.ndarray, alpha_bars: np.ndarray) -> np.ndarray:
    """Closed-form forward diffusion: sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps."""
    raise NotImplementedError
