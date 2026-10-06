# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import numpy as np


def _sqrtm_psd(c: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(c)
    w = np.clip(w, 0.0, None)
    return (v * np.sqrt(w)) @ v.T


def fid(feats_a: np.ndarray, feats_b: np.ndarray) -> float:
    a = np.asarray(feats_a, dtype=np.float64)
    b = np.asarray(feats_b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("both feature sets must be 2-D with the same number of columns")
    if a.shape[0] < 2 or b.shape[0] < 2:
        raise ValueError("each feature set needs at least 2 samples")

    d_mu = a.mean(axis=0) - b.mean(axis=0)
    ca = np.cov(a, rowvar=False, ddof=1)
    cb = np.cov(b, rowvar=False, ddof=1)

    root_a = _sqrtm_psd(ca)
    m = root_a @ cb @ root_a
    m = (m + m.T) / 2.0
    tr_sqrt = float(np.sqrt(np.clip(np.linalg.eigvalsh(m), 0.0, None)).sum())

    value = float(d_mu @ d_mu + np.trace(ca) + np.trace(cb) - 2.0 * tr_sqrt)
    return max(value, 0.0)
