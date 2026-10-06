def make_xy(ids: list[int], block_size: int, i: int) -> tuple[list[int], list[int]]:
    """Return (ids[i:i+block_size], ids[i+1:i+block_size+1]); ValueError if the target runs past the end."""
    raise NotImplementedError


def cross_entropy(logits: list[list[float]], targets: list[int]) -> float:
    """Mean over the batch of logsumexp(row) - row[target], computed with max subtraction."""
    raise NotImplementedError
