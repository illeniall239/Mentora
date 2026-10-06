# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("every embedding must have non-zero norm")
    return x / norms


def _row_ce(logits: np.ndarray) -> float:
    """Mean over rows of -log softmax(row)[true index], via the stable logsumexp form."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=1)) + logits.max(axis=1)
    diag = np.diag(logits)
    return float(np.mean(logsumexp - diag))


def clip_loss(img_embs: np.ndarray, txt_embs: np.ndarray, temperature: float) -> float:
    if img_embs.ndim != 2 or txt_embs.ndim != 2:
        raise ValueError("embeddings must be 2-D (N, d)")
    if img_embs.shape != txt_embs.shape:
        raise ValueError("img_embs and txt_embs must have the same shape")
    if img_embs.shape[0] == 0:
        raise ValueError("the batch must hold at least one pair")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    logits = _normalize(img_embs) @ _normalize(txt_embs).T / temperature
    return 0.5 * (_row_ce(logits) + _row_ce(logits.T))
