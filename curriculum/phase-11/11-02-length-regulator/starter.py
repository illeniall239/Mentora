def durations_to_frames(phonemes: list[str], durs: list[float], hop_s: float) -> list[str]:
    """Expand each phoneme to frames using cumulative half-up rounded boundaries."""
    raise NotImplementedError


def total_frames(durs: list[float], hop_s: float) -> int:
    """round_half_up(sum(durs) / hop_s): the length durations_to_frames must produce."""
    raise NotImplementedError
