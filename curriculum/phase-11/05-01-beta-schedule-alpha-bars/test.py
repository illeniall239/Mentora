import unittest

import numpy as np

from solution import alpha_bars, linear_beta_schedule


class TestLinearBetaSchedule(unittest.TestCase):
    def test_endpoints_are_included(self):
        betas = linear_beta_schedule(5, 0.1, 0.5)
        self.assertEqual(betas.shape, (5,))
        self.assertTrue(np.allclose(betas, [0.1, 0.2, 0.3, 0.4, 0.5], atol=1e-12))
        self.assertTrue(np.allclose(linear_beta_schedule(2, 0.1, 0.5), [0.1, 0.5], atol=1e-12))

    def test_ddpm_defaults(self):
        betas = linear_beta_schedule(1000, 1e-4, 0.02)
        self.assertEqual(betas.shape, (1000,))
        self.assertAlmostEqual(float(betas[0]), 1e-4, places=12)
        self.assertAlmostEqual(float(betas[-1]), 0.02, places=12)
        gaps = np.diff(betas)
        self.assertTrue(np.allclose(gaps, gaps[0], atol=1e-15))
        self.assertTrue(bool(np.all(gaps > 0)))

    def test_constant_beta_when_endpoints_match(self):
        betas = linear_beta_schedule(4, 0.3, 0.3)
        self.assertTrue(np.allclose(betas, 0.3, atol=1e-12))

    def test_bad_arguments_raise(self):
        with self.assertRaises(ValueError):
            linear_beta_schedule(1, 0.1, 0.5)
        with self.assertRaises(ValueError):
            linear_beta_schedule(5, 0.5, 0.1)
        with self.assertRaises(ValueError):
            linear_beta_schedule(5, 0.0, 0.5)
        with self.assertRaises(ValueError):
            linear_beta_schedule(5, 0.1, 1.0)


class TestAlphaBars(unittest.TestCase):
    def test_is_a_cumulative_product_not_a_sum(self):
        got = alpha_bars(np.array([0.1, 0.2]))
        self.assertTrue(np.allclose(got, [0.9, 0.72], atol=1e-12))
        self.assertFalse(np.allclose(got, [0.9, 1.7], atol=1e-3))
        self.assertTrue(np.allclose(alpha_bars(np.array([0.5, 0.5, 0.5])), [0.5, 0.25, 0.125], atol=1e-12))

    def test_first_entry_already_has_one_step_of_noise(self):
        betas = linear_beta_schedule(10, 0.01, 0.2)
        abar = alpha_bars(betas)
        self.assertAlmostEqual(float(abar[0]), 1.0 - float(betas[0]), places=12)
        self.assertNotAlmostEqual(float(abar[0]), 1.0, places=6)

    def test_recurrence_holds(self):
        betas = linear_beta_schedule(50, 1e-3, 0.3)
        abar = alpha_bars(betas)
        self.assertEqual(abar.shape, (50,))
        self.assertTrue(np.allclose(abar[1:], abar[:-1] * (1.0 - betas[1:]), atol=1e-14))

    def test_strictly_decreasing_and_inside_the_unit_interval(self):
        abar = alpha_bars(linear_beta_schedule(1000, 1e-4, 0.02))
        self.assertTrue(bool(np.all(np.diff(abar) < 0)))
        self.assertTrue(bool(np.all((abar > 0.0) & (abar < 1.0))))

    def test_ddpm_schedule_ends_near_zero(self):
        abar = alpha_bars(linear_beta_schedule(1000, 1e-4, 0.02))
        self.assertAlmostEqual(float(abar[0]), 0.9999, places=9)
        self.assertLess(float(abar[-1]), 1e-4)
        self.assertAlmostEqual(float(abar[-1]), 4.0358e-05, delta=1e-8)
        self.assertLess(float(np.sqrt(abar[-1])), 0.01)

    def test_bad_betas_raise(self):
        with self.assertRaises(ValueError):
            alpha_bars(np.array([0.0, 0.5]))
        with self.assertRaises(ValueError):
            alpha_bars(np.array([0.5, 1.0]))
        with self.assertRaises(ValueError):
            alpha_bars(np.array([]))
        with self.assertRaises(ValueError):
            alpha_bars(np.array([[0.1, 0.2]]))

    def test_input_is_not_modified(self):
        betas = linear_beta_schedule(20, 0.01, 0.1)
        copy = betas.copy()
        alpha_bars(betas)
        self.assertTrue(np.array_equal(betas, copy))


if __name__ == "__main__":
    unittest.main(verbosity=2)
