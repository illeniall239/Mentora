import numpy as np


def recall_at_k(sim_matrix: np.ndarray, k: int) -> tuple[float, float]:
    """Recall@k of the diagonal pairs, as (image-to-text, text-to-image)."""
    raise NotImplementedError
