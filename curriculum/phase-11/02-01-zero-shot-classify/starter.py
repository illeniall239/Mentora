import numpy as np


def zero_shot_classify(img_emb: np.ndarray, label_embs: np.ndarray) -> int:
    """Index of the label whose embedding has the largest cosine similarity with the image embedding."""
    raise NotImplementedError
