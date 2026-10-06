def make_mask(n: int, kind: str, prefix_len: int | None = None) -> list[list[bool]]:
    """Build the n x n attention mask for "bidirectional", "causal" or "prefix"; True means keep."""
    raise NotImplementedError
