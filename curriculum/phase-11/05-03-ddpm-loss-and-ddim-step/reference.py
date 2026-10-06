# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import numpy as np


def _check_abar(value: float, name: str) -> float:
    if not (0.0 < float(value) <= 1.0):
        raise ValueError(f"{name} must lie in (0, 1]")
    return float(value)


def _check_pair(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError("the two arrays must have the same shape")


def ddpm_loss(eps: np.ndarray, eps_pred: np.ndarray) -> float:
    _check_pair(eps, eps_pred)
    return float(np.mean((eps - eps_pred) ** 2))


def predict_x0(x_t: np.ndarray, eps_pred: np.ndarray, abar: float) -> np.ndarray:
    _check_pair(x_t, eps_pred)
    a = _check_abar(abar, "abar")
    return (x_t - np.sqrt(1.0 - a) * eps_pred) / np.sqrt(a)


def ddim_step(x_t: np.ndarray, eps_pred: np.ndarray, abar_t: float, abar_prev: float) -> np.ndarray:
    _check_pair(x_t, eps_pred)
    _check_abar(abar_t, "abar_t")
    prev = _check_abar(abar_prev, "abar_prev")
    x0_hat = predict_x0(x_t, eps_pred, abar_t)
    return np.sqrt(prev) * x0_hat + np.sqrt(1.0 - prev) * eps_pred
