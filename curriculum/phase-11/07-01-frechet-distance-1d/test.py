import math
import random
import unittest

from solution import frechet_distance_1d


class TestFrechetDistance1d(unittest.TestCase):
    def test_identical_gaussians_are_zero(self):
        self.assertAlmostEqual(frechet_distance_1d(0.0, 1.0, 0.0, 1.0), 0.0, places=9)
        self.assertAlmostEqual(frechet_distance_1d(-3.5, 7.25, -3.5, 7.25), 0.0, places=9)
        self.assertAlmostEqual(frechet_distance_1d(2.0, 0.0, 2.0, 0.0), 0.0, places=9)

    def test_mean_shift_only(self):
        self.assertAlmostEqual(frechet_distance_1d(0.0, 1.0, 3.0, 1.0), 9.0, places=9)
        self.assertAlmostEqual(frechet_distance_1d(0.0, 0.0, 2.0, 0.0), 4.0, places=9)

    def test_variance_term_uses_variances_not_standard_deviations(self):
        # sigma 1 vs sigma 2: (1 - 2)^2 = 1. Treating the arguments as sigma would give 9.
        self.assertAlmostEqual(frechet_distance_1d(0.0, 1.0, 0.0, 4.0), 1.0, places=9)
        # sigma 1 vs sigma 3
        self.assertAlmostEqual(frechet_distance_1d(0.0, 1.0, 0.0, 9.0), 4.0, places=9)

    def test_both_terms_add(self):
        self.assertAlmostEqual(frechet_distance_1d(0.0, 1.0, 3.0, 4.0), 10.0, places=9)
        self.assertAlmostEqual(frechet_distance_1d(10.0, 9.0, 4.0, 16.0), 36.0 + 1.0, places=9)

    def test_matches_the_closed_form_on_random_gaussians(self):
        rng = random.Random(0)
        for _ in range(200):
            mu1, mu2 = rng.uniform(-50, 50), rng.uniform(-50, 50)
            s1, s2 = rng.uniform(0, 30), rng.uniform(0, 30)
            want = (mu1 - mu2) ** 2 + (math.sqrt(s1) - math.sqrt(s2)) ** 2
            self.assertAlmostEqual(frechet_distance_1d(mu1, s1, mu2, s2), want, places=7)

    def test_symmetric_and_non_negative(self):
        rng = random.Random(1)
        for _ in range(100):
            a = (rng.uniform(-10, 10), rng.uniform(0, 5))
            b = (rng.uniform(-10, 10), rng.uniform(0, 5))
            forward = frechet_distance_1d(a[0], a[1], b[0], b[1])
            backward = frechet_distance_1d(b[0], b[1], a[0], a[1])
            self.assertAlmostEqual(forward, backward, places=9)
            self.assertGreaterEqual(forward, -1e-12)

    def test_cross_term_is_not_dropped(self):
        # forgetting -2*sqrt(s1*s2) would give 5.0 here, and 2.0 for equal variances
        self.assertAlmostEqual(frechet_distance_1d(0.0, 1.0, 0.0, 4.0), 1.0, places=9)
        self.assertAlmostEqual(frechet_distance_1d(0.0, 1.0, 0.0, 1.0), 0.0, places=9)

    def test_negative_variance_raises(self):
        with self.assertRaises(ValueError):
            frechet_distance_1d(0.0, 1.0, 0.0, -1.0)
        with self.assertRaises(ValueError):
            frechet_distance_1d(0.0, -0.5, 0.0, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
