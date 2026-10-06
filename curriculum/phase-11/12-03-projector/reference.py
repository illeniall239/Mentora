# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import math

import numpy as np


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def project(
    features: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray
) -> np.ndarray:
    if features.ndim != 2 or W1.ndim != 2 or W2.ndim != 2 or b1.ndim != 1 or b2.ndim != 1:
        raise ValueError("features, W1 and W2 must be 2-D; b1 and b2 must be 1-D")
    if features.shape[1] != W1.shape[0]:
        raise ValueError("features width must match W1 rows")
    if W1.shape[1] != b1.shape[0] or W1.shape[1] != W2.shape[0]:
        raise ValueError("hidden width must match b1 and W2 rows")
    if W2.shape[1] != b2.shape[0]:
        raise ValueError("output width must match b2")
    return gelu(features @ W1 + b1) @ W2 + b2
