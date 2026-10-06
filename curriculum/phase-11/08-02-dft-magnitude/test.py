import unittest

import numpy as np

from solution import dft_magnitude


def tone(freq, sr, n, phase=0.0, amp=1.0):
    return amp * np.sin(2 * np.pi * freq * np.arange(n) / sr + phase)


class TestDftMagnitude(unittest.TestCase):
    def test_hand_computed_four_point_transforms(self):
        np.testing.assert_allclose(dft_magnitude(np.array([1.0, 1, 1, 1])), [4, 0, 0], atol=1e-9)
        np.testing.assert_allclose(dft_magnitude(np.array([1.0, -1, 1, -1])), [0, 0, 4], atol=1e-9)
        np.testing.assert_allclose(dft_magnitude(np.array([0.0, 1, 0, -1])), [0, 2, 0], atol=1e-9)

    def test_output_length_for_even_and_odd_n(self):
        for n in (4, 5, 16, 17, 64, 65):
            self.assertEqual(dft_magnitude(np.ones(n)).shape, (n // 2 + 1,), msg=f"n={n}")

    def test_matches_a_reference_transform_on_random_signals(self):
        rng = np.random.default_rng(0)
        for n in (16, 31, 64):
            x = rng.normal(size=n)
            np.testing.assert_allclose(dft_magnitude(x), np.abs(np.fft.rfft(x)), rtol=1e-6, atol=1e-8)

    def test_pure_tone_lands_on_the_right_bin(self):
        sr, n, freq = 8000, 400, 1000.0
        out = dft_magnitude(tone(freq, sr, n))
        self.assertEqual(int(np.argmax(out)), 50)
        self.assertAlmostEqual(float(out[50]), n / 2, delta=1e-6)
        self.assertLess(float(np.delete(out, 50).max()), 1e-6)

    def test_dc_bin_is_n_times_the_mean(self):
        x = np.full(32, 0.75)
        out = dft_magnitude(x)
        self.assertAlmostEqual(float(out[0]), 32 * 0.75, places=9)
        self.assertLess(float(out[1:].max()), 1e-9)
        shifted = dft_magnitude(tone(500.0, 8000, 160) + 2.0)
        self.assertAlmostEqual(float(shifted[0]), 2.0 * 160, places=6)

    def test_two_tones_give_two_peaks_with_amplitudes_in_proportion(self):
        sr, n = 8000, 400
        x = tone(1000.0, sr, n) + 0.5 * tone(2000.0, sr, n)
        out = dft_magnitude(x)
        self.assertAlmostEqual(float(out[50]), n / 2, delta=1e-6)
        self.assertAlmostEqual(float(out[100]), 0.5 * n / 2, delta=1e-6)
        rest = np.delete(out, [50, 100])
        self.assertLess(float(rest.max()), 1e-6)

    def test_magnitude_throws_away_the_phase(self):
        sr, n, freq = 8000, 400, 1000.0
        a = dft_magnitude(tone(freq, sr, n))
        b = dft_magnitude(tone(freq, sr, n, phase=np.pi / 2))
        c = dft_magnitude(tone(freq, sr, n, phase=0.7))
        np.testing.assert_allclose(a, b, atol=1e-6)
        np.testing.assert_allclose(a, c, atol=1e-6)

    def test_output_is_real_non_negative_and_bad_input_raises(self):
        rng = np.random.default_rng(1)
        out = dft_magnitude(rng.normal(size=33))
        self.assertFalse(np.iscomplexobj(out))
        self.assertTrue(np.all(out >= 0))
        with self.assertRaises(ValueError):
            dft_magnitude(np.array([]))
        with self.assertRaises(ValueError):
            dft_magnitude(np.ones((4, 4)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
