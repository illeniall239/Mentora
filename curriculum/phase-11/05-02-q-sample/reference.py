# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import numpy as np


def q_sample(x0: np.ndarray, t: int, eps: np.ndarray, alpha_bars: np.ndarray) -> np.ndarray:
    if x0.shape != eps.shape:
        raise ValueError("x0 and eps must have the same shape")
    if alpha_bars.ndim != 1 or alpha_bars.size == 0:
        raise ValueError("alpha_bars must be a non-empty 1-D array")
    if np.any(alpha_bars <= 0.0) or np.any(alpha_bars > 1.0):
        raise ValueError("every alpha_bar must lie in (0, 1]")
    if not isinstance(t, (int, np.integer)) or isinstance(t, bool):
        raise ValueError("t must be an integer index")
    if t < 0 or t >= alpha_bars.size:
        raise ValueError("t must be in [0, len(alpha_bars))")

    abar = float(alpha_bars[t])
    return np.sqrt(abar) * x0 + np.sqrt(1.0 - abar) * eps
