# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import numpy as np


def patch_embed(img: np.ndarray, p: int, W: np.ndarray, pos: np.ndarray, cls_vec: np.ndarray) -> np.ndarray:
    if img.ndim != 3:
        raise ValueError("img must be 3-D (height, width, channels)")
    h, w, c = img.shape
    if p < 1 or h % p or w % p:
        raise ValueError("height and width must be positive multiples of p")
    n = (h // p) * (w // p)
    patch_dim = p * p * c
    if W.ndim != 2 or W.shape[0] != patch_dim:
        raise ValueError("W must have shape (p*p*channels, d)")
    d = W.shape[1]
    if pos.shape != (n + 1, d):
        raise ValueError("pos must have shape (N+1, d)")
    if cls_vec.shape != (d,):
        raise ValueError("cls_vec must have shape (d,)")

    # (h/p, p, w/p, p, c) -> (h/p, w/p, p, p, c) -> (N, p*p*c): row-major over patches and inside each patch.
    patches = img.reshape(h // p, p, w // p, p, c).transpose(0, 2, 1, 3, 4).reshape(n, patch_dim)
    tokens = patches @ W
    return np.concatenate([cls_vec.reshape(1, d), tokens], axis=0) + pos
