def flatten_streams(indices: list[list[int]]) -> list[int]:
    """Interleave Q codebook streams frame-major: seq[f*Q + q] = indices[q][f]."""
    raise NotImplementedError


def unflatten_streams(seq: list[int], n_codebooks: int) -> list[list[int]]:
    """Inverse of flatten_streams: split one LM sequence back into n_codebooks streams."""
    raise NotImplementedError
