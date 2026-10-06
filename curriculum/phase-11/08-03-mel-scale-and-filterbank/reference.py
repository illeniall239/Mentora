# Reference solution: only used by scripts/verify-exercises.mjs. Shown in the Breakdown only after the Learner's own solution passes.
import numpy as np


def hz_to_mel(hz: float) -> float:
    if np.any(np.asarray(hz) < 0):
        raise ValueError("hz must be non-negative")
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: float) -> float:
    if np.any(np.asarray(mel) < 0):
        raise ValueError("mel must be non-negative")
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(n_fft: int, sr: int, n_mels: int) -> np.ndarray:
    if n_fft < 2 or sr < 1 or n_mels < 1:
        raise ValueError("need n_fft >= 2, sr >= 1 and n_mels >= 1")
    n_bins = n_fft // 2 + 1
    bin_hz = np.linspace(0.0, sr / 2.0, n_bins)
    pts = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2.0), n_mels + 2))

    fb = np.zeros((n_mels, n_bins))
    for m in range(n_mels):
        left, center, right = pts[m], pts[m + 1], pts[m + 2]
        rising = (bin_hz - left) / (center - left)
        falling = (right - bin_hz) / (right - center)
        fb[m] = np.maximum(0.0, np.minimum(rising, falling))
    return fb
