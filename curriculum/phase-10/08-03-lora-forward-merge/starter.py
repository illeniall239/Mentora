import numpy as np


def init_lora(d_in: int, d_out: int, r: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (A, B): A = rng.standard_normal((r, d_out)) * 0.02, B = zeros((d_in, r))."""
    raise NotImplementedError


def lora_forward(x: np.ndarray, W: np.ndarray, A: np.ndarray, B: np.ndarray, alpha: float, r: int) -> np.ndarray:
    """x @ W + (alpha / r) * (x @ B) @ A, without forming B @ A."""
    raise NotImplementedError


def merge_lora(W: np.ndarray, A: np.ndarray, B: np.ndarray, alpha: float, r: int) -> np.ndarray:
    """A new matrix W + (alpha / r) * B @ A; W is left untouched."""
    raise NotImplementedError
