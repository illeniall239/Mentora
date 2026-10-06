import unittest

from solution import expected_calibration_error as ece


class TestExpectedCalibrationError(unittest.TestCase):
    def test_perfectly_calibrated_is_zero(self):
        self.assertAlmostEqual(ece([0.5, 0.5, 0.5, 0.5], [True, False, True, False], 10), 0.0, places=9)
        confs = [0.9] * 10 + [0.3] * 10
        right = [True] * 9 + [False] + [True] * 3 + [False] * 7
        self.assertAlmostEqual(ece(confs, right, 10), 0.0, places=9)
        self.assertAlmostEqual(ece([1.0, 1.0], [True, True], 5), 0.0, places=9)

    def test_overconfident_and_underconfident(self):
        self.assertAlmostEqual(ece([0.9, 0.9, 0.9], [False, False, False], 10), 0.9, places=9)
        self.assertAlmostEqual(ece([0.2, 0.2], [True, True], 10), 0.8, places=9)

    def test_bins_are_weighted_by_size(self):
        got = ece([0.8, 0.8, 0.8, 0.2], [False, False, False, False], 2)
        self.assertAlmostEqual(got, 0.75 * 0.8 + 0.25 * 0.2, places=9)
        self.assertNotAlmostEqual(got, 0.5, places=3)

    def test_bins_are_left_open_right_closed(self):
        got = ece([0.5, 0.5, 0.6, 0.6], [True, True, False, False], 4)
        self.assertAlmostEqual(got, 0.5 * 0.5 + 0.5 * 0.6, places=9)
        self.assertAlmostEqual(ece([0.25, 0.26], [True, False], 4), 0.5 * 0.75 + 0.5 * 0.26, places=9)

    def test_endpoints_zero_and_one_are_binned(self):
        self.assertAlmostEqual(ece([0.0, 1.0], [False, True], 10), 0.0, places=9)
        self.assertAlmostEqual(ece([0.0, 1.0], [True, False], 10), 1.0, places=9)
        self.assertAlmostEqual(ece([1.0], [False], 1), 1.0, places=9)

    def test_single_bin_is_mean_gap(self):
        confs = [0.1, 0.4, 0.7, 0.8]
        right = [False, True, True, False]
        self.assertAlmostEqual(ece(confs, right, 1), abs(0.5 - 0.5), places=9)
        self.assertAlmostEqual(ece(confs, [True] * 4, 1), 1.0 - 0.5, places=9)

    def test_empty_bins_are_skipped(self):
        self.assertAlmostEqual(ece([0.95, 0.95], [True, False], 100), 0.45, places=9)

    def test_validation(self):
        with self.assertRaises(ValueError):
            ece([0.3], [True, False], 10)
        with self.assertRaises(ValueError):
            ece([], [], 10)
        with self.assertRaises(ValueError):
            ece([0.3], [True], 0)
        with self.assertRaises(ValueError):
            ece([1.3], [True], 10)
        with self.assertRaises(ValueError):
            ece([-0.1], [True], 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
