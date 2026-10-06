def rvq_encode(vec: list[float], codebooks: list[list[list[float]]]) -> list[int]:
    """One index per codebook: each stage takes the nearest entry (squared L2) to the running residual."""
    raise NotImplementedError


def rvq_decode(indices: list[int], codebooks: list[list[list[float]]]) -> list[float]:
    """Element-wise sum of codebooks[k][indices[k]] over all stages k."""
    raise NotImplementedError


def rvq_errors(vec: list[float], codebooks: list[list[list[float]]]) -> list[float]:
    """L2 norm of the residual after 0, 1, ..., len(codebooks) stages (len(codebooks)+1 values)."""
    raise NotImplementedError
