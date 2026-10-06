def interleave(
    text_ids: list[int], image_slots: list[int], image_token_id: int, n_per_image: int
) -> tuple[list[int], list[tuple[int, int]]]:
    """Insert n_per_image placeholder ids before each slot; return the sequence and each image's (start, end)."""
    raise NotImplementedError
