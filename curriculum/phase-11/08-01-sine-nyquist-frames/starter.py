import numpy as np


def sine_wave(freq: float, sr: int, dur: float) -> np.ndarray:
    """round(sr*dur) samples of sin(2*pi*freq*i/sr), amplitude 1, zero phase."""
    raise NotImplementedError


def nyquist(sr: int) -> float:
    """Highest representable frequency at sampling rate sr."""
    raise NotImplementedError


def frames(samples: int, win: int, hop: int) -> int:
    """Number of complete windows of length win at stride hop, without padding."""
    raise NotImplementedError
