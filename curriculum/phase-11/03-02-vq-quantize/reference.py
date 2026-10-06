# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import numpy as np


def vq_quantize(vectors: np.ndarray, codebook: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if vectors.ndim != 2 or codebook.ndim != 2:
        raise ValueError("vectors and codebook must be 2-D")
    if vectors.shape[1] != codebook.shape[1]:
        raise ValueError("vectors and codebook must have the same width")
    if vectors.shape[0] == 0 or codebook.shape[0] == 0:
        raise ValueError("vectors and codebook must be non-empty")

    # (N, 1, D) - (1, K, D) -> (N, K, D) -> squared distances (N, K)
    diff = vectors[:, None, :] - codebook[None, :, :]
    sq_dist = np.einsum("nkd,nkd->nk", diff, diff)
    indices = np.argmin(sq_dist, axis=1)  # argmin already returns the first minimum
    return indices, codebook[indices].copy()
