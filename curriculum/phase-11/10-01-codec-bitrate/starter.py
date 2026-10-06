def codec_bitrate(frame_rate: float, n_codebooks: int, codebook_size: int) -> float:
    """Bits per second: frame_rate * n_codebooks * log2(codebook_size)."""
    raise NotImplementedError


def tokens_per_second(frame_rate: float, n_codebooks: int) -> float:
    """Tokens an LM must emit per second of audio: frame_rate * n_codebooks."""
    raise NotImplementedError


def compression_ratio(sample_rate: int, bit_depth: int, bitrate_bps: float) -> float:
    """Raw PCM bits per second (sample_rate * bit_depth) divided by the codec bitrate."""
    raise NotImplementedError
