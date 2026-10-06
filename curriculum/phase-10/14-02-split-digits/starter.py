def split_digits(text: str) -> list[str]:
    """Split text into chunks, digit runs cut into groups of at most 3 counted from the right."""
    raise NotImplementedError


def token_count_delta(text: str, merges: dict[tuple[int, int], int]) -> int:
    """Tokens when each split_digits chunk is encoded alone, minus tokens when text is encoded whole."""
    raise NotImplementedError
