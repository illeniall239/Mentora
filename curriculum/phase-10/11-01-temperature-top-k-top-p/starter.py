def apply_temperature(logits: list[float], T: float) -> list[float]:
    """softmax(logits / T) with the max subtracted first; ValueError if T <= 0."""
    raise NotImplementedError


def top_k_filter(probs: list[float], k: int) -> list[float]:
    """Keep the k largest (ties: lower index), zero the rest, renormalize; ValueError if k < 1."""
    raise NotImplementedError


def top_p_filter(probs: list[float], p: float) -> list[float]:
    """Keep the smallest descending-sorted prefix with cumulative >= p, zero the rest, renormalize."""
    raise NotImplementedError
