import math
import unittest

import numpy as np

from solution import d_loss, g_loss_nonsat


class TestGanLosses(unittest.TestCase):
    def test_equilibrium_values(self):
        half = np.full(4, 0.5)
        self.assertAlmostEqual(d_loss(half, half), 2.0 * math.log(2.0), places=9)
        self.assertAlmostEqual(g_loss_nonsat(half), math.log(2.0), places=9)

    def test_matches_the_formula_on_random_batches(self):
        rng = np.random.default_rng(0)
        for n, m in ((1, 1), (5, 9), (64, 64)):
            real = rng.uniform(0.01, 0.99, size=n)
            fake = rng.uniform(0.01, 0.99, size=m)
            want = -np.mean(np.log(real)) - np.mean(np.log(1.0 - fake))
            self.assertAlmostEqual(d_loss(real, fake), float(want), places=9)
            self.assertAlmostEqual(g_loss_nonsat(fake), float(-np.mean(np.log(fake))), places=9)

    def test_two_separate_means_not_one(self):
        # Repeating a batch must not change its mean, so unequal batch sizes are fine.
        self.assertAlmostEqual(d_loss(np.full(2, 0.5), np.full(7, 0.5)), 2.0 * math.log(2.0), places=9)
        real, fake = np.array([0.8, 0.6]), np.array([0.2])
        self.assertAlmostEqual(d_loss(real, fake), d_loss(np.tile(real, 5), np.tile(fake, 3)), places=9)

    def test_discriminator_loss_falls_as_d_improves(self):
        losses = [
            d_loss(np.array([r]), np.array([f]))
            for r, f in ((0.5, 0.5), (0.7, 0.3), (0.9, 0.1), (0.99, 0.01))
        ]
        self.assertEqual(losses, sorted(losses, reverse=True))
        self.assertLess(d_loss(np.ones(3), np.zeros(3)), 1e-9)
        # Swapping the two arguments is a different (and much worse) number.
        self.assertLess(d_loss(np.array([0.9]), np.array([0.1])), d_loss(np.array([0.1]), np.array([0.9])))

    def test_generator_loss_is_non_saturating(self):
        # Near d_fake = 0 the non-saturating loss explodes; the saturating form would sit near 0.
        self.assertGreater(g_loss_nonsat(np.array([0.01])), 4.0)
        self.assertGreater(g_loss_nonsat(np.zeros(3)), 20.0)
        losses = [g_loss_nonsat(np.array([p])) for p in (0.01, 0.1, 0.5, 0.9, 0.999)]
        self.assertEqual(losses, sorted(losses, reverse=True))
        self.assertLess(g_loss_nonsat(np.array([0.999])), 0.01)

    def test_clamping_keeps_results_finite(self):
        for value in (d_loss(np.zeros(2), np.ones(2)), g_loss_nonsat(np.zeros(2)), d_loss(np.ones(2), np.zeros(2))):
            self.assertTrue(math.isfinite(value))
        self.assertGreater(d_loss(np.zeros(2), np.ones(2)), 40.0)

    def test_returns_plain_floats(self):
        self.assertIsInstance(d_loss(np.full(3, 0.5), np.full(3, 0.5)), float)
        self.assertIsInstance(g_loss_nonsat(np.full(3, 0.5)), float)

    def test_bad_inputs_raise(self):
        ok = np.full(2, 0.5)
        with self.assertRaises(ValueError):
            d_loss(np.array([1.5]), ok)
        with self.assertRaises(ValueError):
            d_loss(ok, np.array([-0.1]))
        with self.assertRaises(ValueError):
            d_loss(np.array([]), ok)
        with self.assertRaises(ValueError):
            d_loss(np.full((2, 2), 0.5), ok)
        with self.assertRaises(ValueError):
            g_loss_nonsat(np.array([]))
        with self.assertRaises(ValueError):
            g_loss_nonsat(np.array([2.0]))

    def test_inputs_are_not_modified(self):
        real, fake = np.array([0.0, 1.0, 0.5]), np.array([0.0, 1.0])
        real_copy, fake_copy = real.copy(), fake.copy()
        d_loss(real, fake)
        g_loss_nonsat(fake)
        self.assertTrue(np.array_equal(real, real_copy))
        self.assertTrue(np.array_equal(fake, fake_copy))


if __name__ == "__main__":
    unittest.main(verbosity=2)
