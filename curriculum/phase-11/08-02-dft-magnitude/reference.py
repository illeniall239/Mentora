# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import numpy as np


def dft_magnitude(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("samples must be 1-D")
    n = x.size
    if n == 0:
        raise ValueError("samples must not be empty")
    k = np.arange(n // 2 + 1)
    basis = np.exp(-2j * np.pi * np.outer(k, np.arange(n)) / n)
    return np.abs(basis @ x)
