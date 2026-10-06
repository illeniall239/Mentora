# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import numpy as np


def zero_shot_classify(img_emb: np.ndarray, label_embs: np.ndarray) -> int:
    if img_emb.ndim != 1:
        raise ValueError("img_emb must be 1-D (d,)")
    if label_embs.ndim != 2:
        raise ValueError("label_embs must be 2-D (K, d)")
    if label_embs.shape[0] == 0:
        raise ValueError("label_embs must hold at least one label")
    if label_embs.shape[1] != img_emb.shape[0]:
        raise ValueError("img_emb and label_embs must have the same width")

    img_norm = np.linalg.norm(img_emb)
    label_norms = np.linalg.norm(label_embs, axis=1, keepdims=True)
    if img_norm == 0.0 or np.any(label_norms == 0.0):
        raise ValueError("embeddings must have non-zero norm")

    scores = (label_embs / label_norms) @ (img_emb / img_norm)
    return int(np.argmax(scores))
