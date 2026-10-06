def sinusoidal_pe(pos: int, d_model: int) -> list[float]:
    """The sinusoidal positional encoding for one position: sine at even indices, cosine at odd."""
    raise NotImplementedError


def apply_rope(vec: list[float], pos: int) -> list[float]:
    """Rotate each adjacent pair of vec in its own plane by pos * inv_freq for that pair."""
    raise NotImplementedError
