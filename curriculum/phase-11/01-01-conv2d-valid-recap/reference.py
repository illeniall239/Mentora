# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import numpy as np


def conv2d_valid(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    h, w = img.shape
    kh, kw = kernel.shape
    if kh > h or kw > w:
        raise ValueError("kernel larger than image")
    out = np.empty((h - kh + 1, w - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(img[i : i + kh, j : j + kw] * kernel)
    return out
