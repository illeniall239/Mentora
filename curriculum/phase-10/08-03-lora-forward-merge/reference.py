# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import numpy as np


def _check(W: np.ndarray, A: np.ndarray, B: np.ndarray, r: int) -> None:
    if r < 1:
        raise ValueError("rank must be at least 1")
    if B.shape[1] != r or A.shape[0] != r:
        raise ValueError("B must be (d_in, r) and A must be (r, d_out)")
    if B.shape[0] != W.shape[0] or A.shape[1] != W.shape[1]:
        raise ValueError("B @ A must have the shape of W")


def init_lora(d_in: int, d_out: int, r: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if r < 1:
        raise ValueError("rank must be at least 1")
    A = rng.standard_normal((r, d_out)) * 0.02
    B = np.zeros((d_in, r))  # zero so the adapted layer starts exactly at the pretrained one
    return A, B


def lora_forward(x: np.ndarray, W: np.ndarray, A: np.ndarray, B: np.ndarray, alpha: float, r: int) -> np.ndarray:
    _check(W, A, B, r)
    return x @ W + (alpha / r) * ((x @ B) @ A)  # (n, r) bottleneck, never a (d_in, d_out) product


def merge_lora(W: np.ndarray, A: np.ndarray, B: np.ndarray, alpha: float, r: int) -> np.ndarray:
    _check(W, A, B, r)
    return W + (alpha / r) * (B @ A)
