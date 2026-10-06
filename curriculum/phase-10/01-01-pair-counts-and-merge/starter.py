def get_pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
    """Count adjacent pairs, keys in first-seen order. Must not change ids."""
    raise NotImplementedError


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Return a new list with every non-overlapping occurrence of pair replaced by new_id."""
    raise NotImplementedError
