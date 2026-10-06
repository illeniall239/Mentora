# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import numpy as np


def linear_beta_schedule(T: int, b0: float, b1: float) -> np.ndarray:
    if T < 2:
        raise ValueError("T must be at least 2")
    if b0 <= 0.0 or b1 >= 1.0 or b0 > b1:
        raise ValueError("betas must satisfy 0 < b0 <= b1 < 1")
    return np.linspace(b0, b1, T, dtype=np.float64)


def alpha_bars(betas: np.ndarray) -> np.ndarray:
    if betas.ndim != 1:
        raise ValueError("betas must be 1-D")
    if betas.size == 0:
        raise ValueError("betas must be non-empty")
    if np.any(betas <= 0.0) or np.any(betas >= 1.0):
        raise ValueError("every beta must lie in (0, 1)")
    return np.cumprod(1.0 - betas)
