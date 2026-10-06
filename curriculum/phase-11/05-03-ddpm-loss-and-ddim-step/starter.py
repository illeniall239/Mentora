import numpy as np


def ddpm_loss(eps: np.ndarray, eps_pred: np.ndarray) -> float:
    """L_simple: mean squared error between the true noise and the predicted noise."""
    raise NotImplementedError


def predict_x0(x_t: np.ndarray, eps_pred: np.ndarray, abar: float) -> np.ndarray:
    """Solve the closed-form forward step for x_0 given x_t and a noise prediction."""
    raise NotImplementedError


def ddim_step(x_t: np.ndarray, eps_pred: np.ndarray, abar_t: float, abar_prev: float) -> np.ndarray:
    """One deterministic DDIM update: estimate x_0, then re-noise it to the abar_prev level."""
    raise NotImplementedError
