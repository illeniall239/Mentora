import numpy as np


def gelu(x: np.ndarray) -> np.ndarray:
    """Elementwise tanh-approximation GELU: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x**3)))."""
    raise NotImplementedError


def project(
    features: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray
) -> np.ndarray:
    """Two-layer MLP projector: gelu(features @ W1 + b1) @ W2 + b2, shape (N, d_llm)."""
    raise NotImplementedError
