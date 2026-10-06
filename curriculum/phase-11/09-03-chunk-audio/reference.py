# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
def chunk_audio(n_samples: int, sr: int, chunk_s: float, overlap_s: float) -> list[tuple[int, int]]:
    if n_samples < 0 or sr < 1 or chunk_s <= 0 or overlap_s < 0 or overlap_s >= chunk_s:
        raise ValueError("need n_samples >= 0, sr >= 1 and 0 <= overlap_s < chunk_s")

    chunk = int(round(chunk_s * sr))
    overlap = int(round(overlap_s * sr))
    stride = chunk - overlap
    if chunk < 1 or stride < 1:
        raise ValueError("chunk and stride must be at least one sample")

    windows = []
    start = 0
    while start < n_samples:
        end = min(start + chunk, n_samples)
        windows.append((start, end))
        if end == n_samples:
            break
        start += stride
    return windows
