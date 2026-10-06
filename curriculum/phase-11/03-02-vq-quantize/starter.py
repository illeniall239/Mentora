import numpy as np


def vq_quantize(vectors: np.ndarray, codebook: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Snap each vector to its nearest codebook entry; return the indices and the quantized vectors."""
    raise NotImplementedError
