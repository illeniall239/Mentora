import math
import unittest

import numpy as np

from solution import hz_to_mel, mel_filterbank, mel_to_hz


def expected_filterbank(n_fft, sr, n_mels):
    """Independent construction: mel points from the HTK formula, triangles filled in by loop."""
    to_mel = lambda f: 2595.0 * math.log10(1.0 + f / 700.0)
    to_hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    n_bins = n_fft // 2 + 1
    pts = [to_hz(m) for m in np.linspace(to_mel(0.0), to_mel(sr / 2.0), n_mels + 2)]
    fb = np.zeros((n_mels, n_bins))
    for m in range(n_mels):
        left, center, right = pts[m], pts[m + 1], pts[m + 2]
        for b in range(n_bins):
            f = b * (sr / 2.0) / (n_bins - 1)
            if left < f < center:
                fb[m, b] = (f - left) / (center - left)
            elif center <= f < right:
                fb[m, b] = (right - f) / (right - center)
    return fb


class TestMelFilterbank(unittest.TestCase):
    def test_htk_values_and_the_1000_hz_anchor(self):
        self.assertAlmostEqual(float(hz_to_mel(0.0)), 0.0, places=12)
        self.assertAlmostEqual(float(hz_to_mel(1000.0)), 999.9855371396244, places=6)
        self.assertAlmostEqual(float(hz_to_mel(8000.0)), 2840.023046708319, places=6)
        self.assertAlmostEqual(float(mel_to_hz(0.0)), 0.0, places=12)
        self.assertAlmostEqual(float(mel_to_hz(1000.0)), 1000.0, delta=0.05)

    def test_round_trip(self):
        for hz in (0.0, 1.0, 100.0, 440.0, 1000.0, 7999.5, 22050.0):
            self.assertAlmostEqual(float(mel_to_hz(hz_to_mel(hz))), hz, places=6, msg=f"hz={hz}")
        for mel in (0.0, 150.0, 1000.0, 2840.0):
            self.assertAlmostEqual(float(hz_to_mel(mel_to_hz(mel))), mel, places=6, msg=f"mel={mel}")

    def test_scale_is_near_linear_low_and_compressive_high(self):
        low = float(hz_to_mel(200.0)) / float(hz_to_mel(100.0))
        self.assertGreater(low, 1.8)
        self.assertLess(low, 2.0)
        slope_low = float(hz_to_mel(100.0)) - float(hz_to_mel(0.0))
        slope_high = float(hz_to_mel(8100.0)) - float(hz_to_mel(8000.0))
        self.assertGreater(slope_low, 5 * slope_high)
        with self.assertRaises(ValueError):
            hz_to_mel(-1.0)
        with self.assertRaises(ValueError):
            mel_to_hz(-1.0)

    def test_filterbank_matches_the_triangles(self):
        for n_fft, sr, n_mels in [(512, 16000, 10), (64, 8000, 4), (256, 22050, 23)]:
            got = mel_filterbank(n_fft, sr, n_mels)
            self.assertEqual(got.shape, (n_mels, n_fft // 2 + 1))
            np.testing.assert_allclose(got, expected_filterbank(n_fft, sr, n_mels), atol=1e-9)

    def test_weights_are_bounded_and_peaks_move_right(self):
        fb = mel_filterbank(512, 16000, 20)
        self.assertGreaterEqual(float(fb.min()), 0.0)
        self.assertLessEqual(float(fb.max()), 1.0 + 1e-12)
        self.assertGreater(float(fb.max()), 0.99)
        peaks = [int(np.argmax(fb[m])) for m in range(20)]
        self.assertEqual(peaks, sorted(peaks))
        self.assertEqual(len(set(peaks)), 20)
        self.assertAlmostEqual(float(fb[0, 0]), 0.0, places=12)

    def test_filters_widen_with_frequency(self):
        fb = mel_filterbank(512, 16000, 20)
        widths = [int((fb[m] > 0).sum()) for m in range(20)]
        self.assertGreater(widths[-1], 3 * widths[0])
        self.assertGreater(widths[-1], widths[10])

    def test_points_are_even_in_mel_not_in_hertz(self):
        fb = mel_filterbank(1024, 16000, 12)
        bin_hz = np.linspace(0.0, 8000.0, 513)
        peaks_hz = np.array([bin_hz[int(np.argmax(fb[m]))] for m in range(12)])
        mel_gaps = np.diff([float(hz_to_mel(f)) for f in peaks_hz])
        hz_gaps = np.diff(peaks_hz)
        self.assertLess(float(np.std(mel_gaps)), 8.0)
        self.assertGreater(float(hz_gaps[-1]), 2 * float(hz_gaps[0]))

    def test_neighbouring_filters_overlap_and_a_spectrum_can_be_projected(self):
        fb = mel_filterbank(512, 16000, 10)
        overlap = np.logical_and(fb[3] > 0, fb[4] > 0).sum()
        self.assertGreater(overlap, 0)
        spectrum = np.zeros(257)
        spectrum[int(np.argmax(fb[5]))] = 2.0
        mel_spec = fb @ spectrum
        self.assertEqual(mel_spec.shape, (10,))
        self.assertEqual(int(np.argmax(mel_spec)), 5)
        self.assertAlmostEqual(float(mel_spec[5]), 2.0 * float(fb[5].max()), places=9)

    def test_invalid_arguments_raise(self):
        for args in [(1, 16000, 10), (512, 0, 10), (512, 16000, 0), (512, 16000, -3)]:
            with self.assertRaises(ValueError, msg=f"mel_filterbank{args}"):
                mel_filterbank(*args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
