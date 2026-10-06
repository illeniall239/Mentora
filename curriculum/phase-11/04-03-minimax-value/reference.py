# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import numpy as np


def _check(p_data: np.ndarray, p_g: np.ndarray) -> None:
    for name, p in (("p_data", p_data), ("p_g", p_g)):
        if p.ndim != 1:
            raise ValueError(f"{name} must be 1-D")
        if p.size == 0:
            raise ValueError(f"{name} must be non-empty")
        if np.any(p < 0.0):
            raise ValueError(f"{name} must be non-negative")
        if abs(float(p.sum()) - 1.0) > 1e-8:
            raise ValueError(f"{name} must sum to 1")
    if p_data.shape != p_g.shape:
        raise ValueError("p_data and p_g must share the same support")


def optimal_d(p_data: np.ndarray, p_g: np.ndarray) -> np.ndarray:
    _check(p_data, p_g)
    total = p_data + p_g
    return np.where(total > 0.0, p_data / np.where(total > 0.0, total, 1.0), 0.5)


def minimax_value(p_data: np.ndarray, p_g: np.ndarray) -> float:
    _check(p_data, p_g)
    d = optimal_d(p_data, p_g)
    # 0 * log(0) = 0: only terms with positive weight contribute.
    real = np.sum(p_data[p_data > 0.0] * np.log(d[p_data > 0.0]))
    fake = np.sum(p_g[p_g > 0.0] * np.log(1.0 - d[p_g > 0.0]))
    return float(real + fake)
