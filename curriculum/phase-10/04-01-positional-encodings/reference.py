# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import math


def _inv_freqs(d_model: int, pos: int) -> list[float]:
    if not isinstance(d_model, int) or d_model <= 0 or d_model % 2 != 0:
        raise ValueError("d_model must be a positive even number")
    if pos < 0:
        raise ValueError("pos must be >= 0")
    return [1.0 / (10000.0 ** (2 * i / d_model)) for i in range(d_model // 2)]


def sinusoidal_pe(pos: int, d_model: int) -> list[float]:
    pe: list[float] = []
    for inv_freq in _inv_freqs(d_model, pos):
        angle = pos * inv_freq
        pe.append(math.sin(angle))
        pe.append(math.cos(angle))
    return pe


def apply_rope(vec: list[float], pos: int) -> list[float]:
    out: list[float] = []
    for i, inv_freq in enumerate(_inv_freqs(len(vec), pos)):
        theta = pos * inv_freq
        c, s = math.cos(theta), math.sin(theta)
        x, y = vec[2 * i], vec[2 * i + 1]
        out.append(x * c - y * s)
        out.append(x * s + y * c)
    return out
