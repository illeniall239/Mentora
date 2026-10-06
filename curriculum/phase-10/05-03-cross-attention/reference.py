# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
from collections.abc import Sequence

import numpy as np


def cross_attention(
    dec_x: np.ndarray,
    enc_out: np.ndarray,
    params: dict,
    enc_mask: Sequence[bool] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if dec_x.ndim != 2 or enc_out.ndim != 2:
        raise ValueError("dec_x and enc_out must both be 2-D")
    if dec_x.shape[1] != enc_out.shape[1]:
        raise ValueError("dec_x and enc_out must share d_model")
    m = enc_out.shape[0]

    q = dec_x @ params["wq"]  # (n, d_k) -- from the decoder
    k = enc_out @ params["wk"]  # (m, d_k) -- from the encoder
    v = enc_out @ params["wv"]  # (m, d_v) -- from the encoder

    scores = q @ k.T / np.sqrt(q.shape[1])
    if enc_mask is not None:
        keep = np.asarray(enc_mask, dtype=bool)
        if keep.shape != (m,):
            raise ValueError("enc_mask must have one entry per encoder position")
        if not keep.any():
            raise ValueError("enc_mask blocks every encoder position")
        scores = np.where(keep[None, :], scores, -np.inf)

    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    weights = exps / np.sum(exps, axis=-1, keepdims=True)
    return (weights @ v) @ params["wo"], weights
