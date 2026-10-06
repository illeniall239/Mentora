def chunk_audio(n_samples: int, sr: int, chunk_s: float, overlap_s: float) -> list[tuple[int, int]]:
    """Overlapping (start, end) sample ranges covering the whole clip; end is exclusive."""
    raise NotImplementedError
