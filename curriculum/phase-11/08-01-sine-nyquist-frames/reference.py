# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import numpy as np


def sine_wave(freq: float, sr: int, dur: float) -> np.ndarray:
    if sr < 1:
        raise ValueError("sr must be at least 1")
    if dur < 0:
        raise ValueError("dur must be non-negative")
    n = int(round(sr * dur))
    return np.sin(2.0 * np.pi * freq * np.arange(n) / sr)


def nyquist(sr: int) -> float:
    if sr < 1:
        raise ValueError("sr must be at least 1")
    return sr / 2.0


def frames(samples: int, win: int, hop: int) -> int:
    if samples < 0 or win < 1 or hop < 1:
        raise ValueError("samples must be non-negative and win, hop at least 1")
    if samples < win:
        return 0
    return 1 + (samples - win) // hop
