import numpy as np


def optimal_d(p_data: np.ndarray, p_g: np.ndarray) -> np.ndarray:
    """The optimal discriminator p_data / (p_data + p_g), elementwise, with 0.5 where both are zero."""
    raise NotImplementedError


def minimax_value(p_data: np.ndarray, p_g: np.ndarray) -> float:
    """The minimax objective evaluated at the optimal discriminator, with 0 * log(0) = 0."""
    raise NotImplementedError
