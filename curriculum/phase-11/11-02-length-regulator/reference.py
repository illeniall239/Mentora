# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import math


def _round_half_up(x: float) -> int:
    return math.floor(x + 0.5)


def _check(durs: list[float], hop_s: float) -> None:
    if hop_s <= 0:
        raise ValueError("hop_s must be positive")
    if any(d < 0 for d in durs):
        raise ValueError("durations must not be negative")


def durations_to_frames(phonemes: list[str], durs: list[float], hop_s: float) -> list[str]:
    if len(phonemes) != len(durs):
        raise ValueError("one duration per phoneme")
    _check(durs, hop_s)
    frames: list[str] = []
    cum = 0.0
    boundary = 0
    for phoneme, dur in zip(phonemes, durs):
        cum += dur
        next_boundary = _round_half_up(cum / hop_s)
        frames.extend([phoneme] * (next_boundary - boundary))
        boundary = next_boundary
    return frames


def total_frames(durs: list[float], hop_s: float) -> int:
    _check(durs, hop_s)
    cum = 0.0
    for d in durs:
        cum += d
    return _round_half_up(cum / hop_s)
