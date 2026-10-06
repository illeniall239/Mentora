def wer(ref: str, hyp: str) -> tuple[int, int, int, int, float]:
    """Word error rate: returns (substitutions, deletions, insertions, reference words, rate)."""
    raise NotImplementedError
