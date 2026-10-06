import numpy as np


def reinforce_grad(logits: np.ndarray, actions: np.ndarray, returns: np.ndarray, baseline: float = 0.0) -> np.ndarray:
    """(n, A) ascent gradient of the mean advantage-weighted log-probability of the sampled actions."""
    raise NotImplementedError
