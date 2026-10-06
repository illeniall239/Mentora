# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
def flatten_streams(indices: list[list[int]]) -> list[int]:
    if not indices:
        raise ValueError("need at least one codebook stream")
    t = len(indices[0])
    if any(len(stream) != t for stream in indices):
        raise ValueError("every codebook stream must have the same number of frames")
    return [stream[f] for f in range(t) for stream in indices]


def unflatten_streams(seq: list[int], n_codebooks: int) -> list[list[int]]:
    if n_codebooks < 1:
        raise ValueError("n_codebooks must be at least 1")
    if len(seq) % n_codebooks:
        raise ValueError("sequence length must be a multiple of n_codebooks")
    return [seq[q::n_codebooks] for q in range(n_codebooks)]
