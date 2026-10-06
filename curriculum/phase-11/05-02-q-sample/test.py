import math
import unittest

import numpy as np

from solution import q_sample

ABARS = np.array([0.64, 0.25])


def ddpm_abars(t_steps: int = 1000) -> np.ndarray:
    betas = np.linspace(1e-4, 0.02, t_steps)
    return np.cumprod(1.0 - betas)


class TestQSample(unittest.TestCase):
    def test_hand_values(self):
        ones = np.ones(2)
        self.assertTrue(np.allclose(q_sample(ones, 0, np.zeros(2), ABARS), [0.8, 0.8], atol=1e-12))
        self.assertTrue(np.allclose(q_sample(np.zeros(2), 0, ones, ABARS), [0.6, 0.6], atol=1e-12))
        self.assertTrue(np.allclose(q_sample(ones, 0, ones, ABARS), [1.4, 1.4], atol=1e-12))
        want = math.sqrt(0.25) + math.sqrt(0.75)
        self.assertTrue(np.allclose(q_sample(ones, 1, ones, ABARS), [want, want], atol=1e-12))

    def test_matches_the_closed_form_on_random_inputs(self):
        rng = np.random.default_rng(0)
        abars = ddpm_abars()
        for shape in ((5,), (3, 4), (2, 4, 4)):
            x0 = rng.normal(size=shape)
            eps = rng.normal(size=shape)
            for t in (0, 1, 137, 999):
                a = float(abars[t])
                want = math.sqrt(a) * x0 + math.sqrt(1.0 - a) * eps
                got = q_sample(x0, t, eps, abars)
                self.assertEqual(got.shape, x0.shape)
                self.assertTrue(np.allclose(got, want, atol=1e-12), f"shape={shape} t={t}")

    def test_the_square_roots_are_on_the_right_quantities(self):
        # Isolating each coefficient: they must be sqrt(abar) and sqrt(1 - abar), not abar and 1 - abar.
        abars = ddpm_abars()
        for t in (0, 10, 500, 999):
            a = float(q_sample(np.ones(1), t, np.zeros(1), abars)[0])
            b = float(q_sample(np.zeros(1), t, np.ones(1), abars)[0])
            self.assertAlmostEqual(a, math.sqrt(abars[t]), places=12)
            self.assertAlmostEqual(b, math.sqrt(1.0 - abars[t]), places=12)
            self.assertAlmostEqual(a * a + b * b, 1.0, places=12)

    def test_t_zero_is_almost_the_clean_image(self):
        rng = np.random.default_rng(1)
        abars = ddpm_abars()
        x0 = rng.normal(size=2000)
        xt = q_sample(x0, 0, rng.normal(size=2000), abars)
        self.assertTrue(np.allclose(xt, x0, atol=0.06))
        self.assertFalse(np.array_equal(xt, x0))

    def test_variance_is_preserved_at_every_t(self):
        rng = np.random.default_rng(2)
        abars = ddpm_abars()
        x0 = rng.normal(size=20000)
        eps = rng.normal(size=20000)
        for t in (0, 50, 300, 700, 999):
            xt = q_sample(x0, t, eps, abars)
            self.assertAlmostEqual(float(xt.var()), 1.0, delta=0.05, msg=f"t={t}")
            self.assertAlmostEqual(float(xt.mean()), 0.0, delta=0.05, msg=f"t={t}")

    def test_the_last_step_is_essentially_pure_noise(self):
        rng = np.random.default_rng(3)
        abars = ddpm_abars()
        x0 = rng.normal(size=20000) * 3.0
        eps = rng.normal(size=20000)
        xt = q_sample(x0, 999, eps, abars)
        self.assertTrue(np.allclose(xt, eps, atol=0.15))
        corr = float(np.corrcoef(xt, x0)[0, 1])
        self.assertLess(abs(corr), 0.05)

    def test_bad_inputs_raise(self):
        x0 = np.ones(3)
        with self.assertRaises(ValueError):
            q_sample(x0, 2, np.ones(3), ABARS)
        with self.assertRaises(ValueError):
            q_sample(x0, -1, np.ones(3), ABARS)
        with self.assertRaises(ValueError):
            q_sample(x0, 0, np.ones(4), ABARS)
        with self.assertRaises(ValueError):
            q_sample(x0, 0, np.ones(3), np.array([]))
        with self.assertRaises(ValueError):
            q_sample(x0, 0, np.ones(3), np.array([[0.5, 0.25]]))
        with self.assertRaises(ValueError):
            q_sample(x0, 0, np.ones(3), np.array([0.0, 0.25]))
        with self.assertRaises(ValueError):
            q_sample(x0, 0, np.ones(3), np.array([1.5, 0.25]))

    def test_inputs_are_not_modified(self):
        x0, eps = np.ones(4), np.full(4, 2.0)
        abars = ABARS.copy()
        q_sample(x0, 1, eps, abars)
        self.assertTrue(np.array_equal(x0, np.ones(4)))
        self.assertTrue(np.array_equal(eps, np.full(4, 2.0)))
        self.assertTrue(np.array_equal(abars, ABARS))


if __name__ == "__main__":
    unittest.main(verbosity=2)
