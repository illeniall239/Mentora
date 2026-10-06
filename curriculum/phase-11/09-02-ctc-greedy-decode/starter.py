def ctc_greedy_decode(frame_ids: list[int], blank: int) -> list[int]:
    """Collapse runs of identical frame ids, then drop blanks, and return the token ids."""
    raise NotImplementedError
