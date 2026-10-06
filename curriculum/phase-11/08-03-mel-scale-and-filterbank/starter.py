import numpy as np


def hz_to_mel(hz: float) -> float:
    """HTK mel scale: 2595 * log10(1 + hz / 700)."""
    raise NotImplementedError


def mel_to_hz(mel: float) -> float:
    """Inverse HTK mel scale: 700 * (10 ** (mel / 2595) - 1)."""
    raise NotImplementedError


def mel_filterbank(n_fft: int, sr: int, n_mels: int) -> np.ndarray:
    """(n_mels, n_fft // 2 + 1) matrix of unit-height triangular mel filters."""
    raise NotImplementedError
