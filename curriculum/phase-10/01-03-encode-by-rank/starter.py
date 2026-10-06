def encode(text: str, merges: dict[tuple[int, int], int]) -> list[int]:
    """UTF-8 bytes of text with merges applied lowest rank first (rank = position in the dict)."""
    raise NotImplementedError
