import unittest

import numpy as np

from solution import frames, nyquist, sine_wave


class TestSineNyquistFrames(unittest.TestCase):
    def test_nyquist(self):
        self.assertAlmostEqual(nyquist(16000), 8000.0, places=9)
        self.assertAlmostEqual(nyquist(44100), 22050.0, places=9)
        self.assertAlmostEqual(nyquist(1), 0.5, places=9)
        with self.assertRaises(ValueError):
            nyquist(0)

    def test_sample_count_and_first_sample(self):
        self.assertEqual(sine_wave(440, 16000, 0.5).shape, (8000,))
        self.assertEqual(sine_wave(440, 16000, 1.0).shape, (16000,))
        self.assertEqual(sine_wave(440, 16000, 0.0).shape, (0,))
        self.assertAlmostEqual(float(sine_wave(440, 16000, 0.01)[0]), 0.0, places=12)

    def test_one_cycle_by_hand(self):
        got = sine_wave(1, 4, 1.0)
        np.testing.assert_allclose(got, [0.0, 1.0, 0.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(sine_wave(0, 1000, 0.01), np.zeros(10), atol=1e-12)

    def test_period_and_amplitude(self):
        sr, freq = 16000, 500.0
        x = sine_wave(freq, sr, 0.1)
        period = int(sr / freq)
        self.assertEqual(period, 32)
        np.testing.assert_allclose(x[:200], x[period : period + 200], atol=1e-9)
        self.assertAlmostEqual(float(x.max()), 1.0, places=6)
        self.assertAlmostEqual(float(x.min()), -1.0, places=6)
        self.assertAlmostEqual(float((x**2).mean()), 0.5, places=3)

    def test_aliasing_above_nyquist(self):
        sr, dur, f = 16000, 0.05, 3000.0
        base = sine_wave(f, sr, dur)
        np.testing.assert_allclose(sine_wave(sr - f, sr, dur), -base, atol=1e-8)
        np.testing.assert_allclose(sine_wave(sr + f, sr, dur), base, atol=1e-8)
        self.assertGreater(sr - f, nyquist(sr))

    def test_frame_counts(self):
        self.assertEqual(frames(16000, 400, 160), 98)
        self.assertEqual(frames(400, 400, 160), 1)
        self.assertEqual(frames(399, 400, 160), 0)
        self.assertEqual(frames(0, 400, 160), 0)
        self.assertEqual(frames(1000, 400, 400), 2)
        self.assertEqual(frames(1000, 1, 1), 1000)

    def test_frames_step_up_exactly_at_each_hop(self):
        for extra in range(0, 5):
            self.assertEqual(frames(400 + extra, 400, 160), 1, msg=f"extra={extra}")
        self.assertEqual(frames(559, 400, 160), 1)
        self.assertEqual(frames(560, 400, 160), 2)
        self.assertEqual(frames(719, 400, 160), 2)
        self.assertEqual(frames(720, 400, 160), 3)

    def test_invalid_arguments_raise(self):
        with self.assertRaises(ValueError):
            sine_wave(440, 0, 1.0)
        with self.assertRaises(ValueError):
            sine_wave(440, 16000, -0.5)
        for args in [(-1, 400, 160), (100, 0, 160), (100, 400, 0), (100, 400, -2)]:
            with self.assertRaises(ValueError, msg=f"frames{args}"):
                frames(*args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
