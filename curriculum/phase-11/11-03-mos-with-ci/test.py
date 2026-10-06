import math
import random
import unittest

from solution import mos


class TestMosWithCi(unittest.TestCase):
    def test_hand_example(self):
        mean, lo, hi = mos([1, 2, 3, 4, 5])
        self.assertAlmostEqual(mean, 3.0, places=9)
        self.assertAlmostEqual(lo, 1.6140707088743669, places=9)
        self.assertAlmostEqual(hi, 4.385929291125633, places=9)

    def test_sample_standard_deviation_not_population(self):
        _, lo, hi = mos([1, 2, 3, 4, 5])
        margin = (hi - lo) / 2
        self.assertAlmostEqual(margin, 1.96 * math.sqrt(2.5) / math.sqrt(5), places=9)
        population_margin = 1.96 * math.sqrt(2.0) / math.sqrt(5)
        self.assertNotAlmostEqual(margin, population_margin, places=3)

    def test_uses_1_96_not_a_rounder_constant(self):
        _, lo, hi = mos([4, 4, 5, 3, 4, 4])
        self.assertAlmostEqual(lo, 3.493930176095564, places=9)
        self.assertAlmostEqual(hi, 4.506069823904436, places=9)

    def test_interval_is_symmetric_around_the_mean(self):
        rng = random.Random(0)
        ratings = [rng.uniform(1.0, 5.0) for _ in range(37)]
        mean, lo, hi = mos(ratings)
        self.assertAlmostEqual(mean, sum(ratings) / len(ratings), places=9)
        self.assertAlmostEqual(hi - mean, mean - lo, places=12)
        self.assertLess(lo, mean)
        self.assertLess(mean, hi)

    def test_interval_narrows_as_ratings_are_added(self):
        _, lo5, hi5 = mos([1, 2, 3, 4, 5])
        _, lo20, hi20 = mos([1, 2, 3, 4, 5] * 4)
        self.assertAlmostEqual(lo20, 2.364092029167411, places=9)
        self.assertAlmostEqual(hi20, 3.635907970832589, places=9)
        self.assertLess(hi20 - lo20, hi5 - lo5)
        self.assertLess(hi20 - lo20, 0.55 * (hi5 - lo5))

    def test_unanimous_ratings_give_a_zero_width_interval(self):
        self.assertEqual(mos([4.0, 4.0, 4.0]), (4.0, 4.0, 4.0))
        mean, lo, hi = mos([5.0, 5.0])
        self.assertAlmostEqual(mean, 5.0, places=9)
        self.assertAlmostEqual(hi - lo, 0.0, places=12)

    def test_validation(self):
        with self.assertRaises(ValueError):
            mos([])
        with self.assertRaises(ValueError):
            mos([4.0])
        with self.assertRaises(ValueError):
            mos([4.0, 6.0])
        with self.assertRaises(ValueError):
            mos([0.5, 4.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
