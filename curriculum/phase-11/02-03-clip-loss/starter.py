import numpy as np


def clip_loss(img_embs: np.ndarray, txt_embs: np.ndarray, temperature: float) -> float:
    """Symmetric InfoNCE over one batch of paired embeddings: normalize, scale by 1/temperature, both directions."""
    raise NotImplementedError
