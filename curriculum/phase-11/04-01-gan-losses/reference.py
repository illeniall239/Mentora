# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import numpy as np

EPS = 1e-12


def _probs(p: np.ndarray, name: str) -> np.ndarray:
    if p.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    if p.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError(f"{name} must hold probabilities in [0, 1]")
    return np.clip(p, EPS, 1.0 - EPS)


def d_loss(d_real: np.ndarray, d_fake: np.ndarray) -> float:
    real = _probs(d_real, "d_real")
    fake = _probs(d_fake, "d_fake")
    return float(-np.mean(np.log(real)) - np.mean(np.log(1.0 - fake)))


def g_loss_nonsat(d_fake: np.ndarray) -> float:
    fake = _probs(d_fake, "d_fake")
    return float(-np.mean(np.log(fake)))
