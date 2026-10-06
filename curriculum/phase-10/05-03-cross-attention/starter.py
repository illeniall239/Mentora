from collections.abc import Sequence

import numpy as np


def cross_attention(
    dec_x: np.ndarray,
    enc_out: np.ndarray,
    params: dict,
    enc_mask: Sequence[bool] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Attend from decoder queries to encoder keys/values; return (output, weights)."""
    raise NotImplementedError
