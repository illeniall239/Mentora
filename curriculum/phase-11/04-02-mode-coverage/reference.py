# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import numpy as np


def mode_coverage(samples: np.ndarray, centers: np.ndarray, radius: float) -> float:
    if samples.ndim != 2 or centers.ndim != 2:
        raise ValueError("samples and centers must be 2-D")
    if samples.shape[1] != centers.shape[1]:
        raise ValueError("samples and centers must have the same width")
    if samples.shape[0] == 0 or centers.shape[0] == 0:
        raise ValueError("samples and centers must be non-empty")
    if radius < 0:
        raise ValueError("radius must be non-negative")

    # (N, 1, D) - (1, M, D) -> distances (N, M)
    diff = samples[:, None, :] - centers[None, :, :]
    dist = np.sqrt(np.einsum("nmd,nmd->nm", diff, diff))
    covered = np.any(dist <= radius, axis=0)  # per mode: reached by anybody?
    return float(np.mean(covered))
