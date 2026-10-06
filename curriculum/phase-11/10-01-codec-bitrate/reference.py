# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import math


def codec_bitrate(frame_rate: float, n_codebooks: int, codebook_size: int) -> float:
    if frame_rate <= 0 or n_codebooks < 1 or codebook_size < 2:
        raise ValueError("need frame_rate > 0, n_codebooks >= 1 and codebook_size >= 2")
    return frame_rate * n_codebooks * math.log2(codebook_size)


def tokens_per_second(frame_rate: float, n_codebooks: int) -> float:
    if frame_rate <= 0 or n_codebooks < 1:
        raise ValueError("need frame_rate > 0 and n_codebooks >= 1")
    return frame_rate * n_codebooks


def compression_ratio(sample_rate: int, bit_depth: int, bitrate_bps: float) -> float:
    if sample_rate < 1 or bit_depth < 1 or bitrate_bps <= 0:
        raise ValueError("need sample_rate >= 1, bit_depth >= 1 and bitrate_bps > 0")
    return (sample_rate * bit_depth) / bitrate_bps
