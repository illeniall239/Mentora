def token_cost(text: str, merges: dict[tuple[int, int], int]) -> int:
    """Number of tokens text becomes: UTF-8 bytes with each merge applied once, in the dict's order."""
    raise NotImplementedError
