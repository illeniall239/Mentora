import math
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from solution import ddim_step, ddpm_loss, predict_x0


def ddpm_abars(t_steps: int = 1000) -> np.ndarray:
    return np.cumprod(1.0 - np.linspace(1e-4, 0.02, t_steps))


def forward(x0: np.ndarray, eps: np.ndarray, abar: float) -> np.ndarray:
    return math.sqrt(abar) * x0 + math.sqrt(1.0 - abar) * eps


class TestDdpmLoss(unittest.TestCase):
    def test_zero_for_a_perfect_prediction(self):
        rng = np.random.default_rng(0)
        eps = rng.normal(size=(4, 5))
        self.assertEqual(ddpm_loss(eps, eps.copy()), 0.0)

    def test_is_a_mean_not_a_sum(self):
        self.assertAlmostEqual(ddpm_loss(np.zeros(2), np.array([1.0, -1.0])), 1.0, places=12)
        self.assertAlmostEqual(ddpm_loss(np.zeros(100), np.ones(100)), 1.0, places=12)
        self.assertAlmostEqual(ddpm_loss(np.zeros((3, 4, 5)), np.full((3, 4, 5), 2.0)), 4.0, places=12)

    def test_matches_torch_mse_and_is_symmetric(self):
        rng = np.random.default_rng(1)
        for shape in ((7,), (3, 4), (2, 3, 4)):
            a, b = rng.normal(size=shape), rng.normal(size=shape)
            want = float(F.mse_loss(torch.tensor(a), torch.tensor(b)))
            self.assertAlmostEqual(ddpm_loss(a, b), want, places=12)
            self.assertAlmostEqual(ddpm_loss(b, a), want, places=12)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            ddpm_loss(np.zeros(3), np.zeros(4))


class TestPredictX0(unittest.TestCase):
    def test_hand_values(self):
        self.assertTrue(np.allclose(predict_x0(np.array([1.4]), np.array([1.0]), 0.64), [1.0], atol=1e-12))
        self.assertTrue(np.allclose(predict_x0(np.array([0.6]), np.array([1.0]), 0.64), [0.0], atol=1e-12))

    def test_inverts_the_forward_process_exactly(self):
        rng = np.random.default_rng(2)
        abars = ddpm_abars()
        x0 = rng.normal(size=(3, 8))
        eps = rng.normal(size=(3, 8))
        for t in (0, 25, 400, 999):
            x_t = forward(x0, eps, float(abars[t]))
            self.assertTrue(np.allclose(predict_x0(x_t, eps, float(abars[t])), x0, atol=1e-10), f"t={t}")

    def test_small_abar_amplifies_prediction_error(self):
        abars = ddpm_abars()
        x_t = np.array([0.5])
        good = predict_x0(x_t, np.array([0.0]), float(abars[999]))
        nudged = predict_x0(x_t, np.array([0.01]), float(abars[999]))
        self.assertGreater(abs(float((nudged - good)[0])), 1.0)
        early = abs(float((predict_x0(x_t, np.array([0.01]), 0.9) - predict_x0(x_t, np.array([0.0]), 0.9))[0]))
        self.assertLess(early, 0.01)

    def test_bad_abar_raises(self):
        with self.assertRaises(ValueError):
            predict_x0(np.ones(2), np.ones(2), 0.0)
        with self.assertRaises(ValueError):
            predict_x0(np.ones(2), np.ones(2), 1.5)
        with self.assertRaises(ValueError):
            predict_x0(np.ones(2), np.ones(3), 0.5)


class TestDdimStep(unittest.TestCase):
    def test_hand_values(self):
        x_t, eps = np.array([1.4]), np.array([1.0])
        self.assertTrue(np.allclose(ddim_step(x_t, eps, 0.64, 0.64), [1.4], atol=1e-12))
        self.assertTrue(np.allclose(ddim_step(x_t, eps, 0.64, 1.0), [1.0], atol=1e-12))
        want = 0.5 * 1.0 + math.sqrt(0.75)
        self.assertTrue(np.allclose(ddim_step(x_t, eps, 0.64, 0.25), [want], atol=1e-12))

    def test_lands_on_the_forward_process_point(self):
        rng = np.random.default_rng(3)
        abars = ddpm_abars()
        x0, eps = rng.normal(size=(2, 6)), rng.normal(size=(2, 6))
        x_t = forward(x0, eps, float(abars[900]))
        for prev in (900, 700, 300, 0):
            got = ddim_step(x_t, eps, float(abars[900]), float(abars[prev]))
            self.assertTrue(np.allclose(got, forward(x0, eps, float(abars[prev])), atol=1e-10), f"prev={prev}")

    def test_is_deterministic(self):
        rng = np.random.default_rng(4)
        x_t, eps = rng.normal(size=50), rng.normal(size=50)
        a = ddim_step(x_t, eps, 0.3, 0.6)
        b = ddim_step(x_t, eps, 0.3, 0.6)
        self.assertTrue(np.array_equal(a, b))

    def test_a_skipping_trajectory_recovers_x0(self):
        rng = np.random.default_rng(5)
        abars = ddpm_abars()
        x0, eps = rng.normal(size=(4, 4)), rng.normal(size=(4, 4))
        x = forward(x0, eps, float(abars[999]))
        schedule = list(range(999, -1, -100)) + [0]
        for t, prev in zip(schedule, schedule[1:]):
            x = ddim_step(x, eps, float(abars[t]), float(abars[prev]))
        x = ddim_step(x, eps, float(abars[0]), 1.0)
        self.assertEqual(len(schedule), 11)  # 11 jumps instead of 1000 steps
        self.assertTrue(np.allclose(x, x0, atol=1e-8))

    def test_bad_abars_raise(self):
        with self.assertRaises(ValueError):
            ddim_step(np.ones(2), np.ones(2), 0.5, 1.5)
        with self.assertRaises(ValueError):
            ddim_step(np.ones(2), np.ones(2), 0.0, 0.5)
        with self.assertRaises(ValueError):
            ddim_step(np.ones(2), np.ones(3), 0.5, 0.5)

    def test_inputs_are_not_modified(self):
        x_t, eps = np.array([1.4, 0.2]), np.array([1.0, -1.0])
        x_copy, eps_copy = x_t.copy(), eps.copy()
        ddpm_loss(eps, eps)
        predict_x0(x_t, eps, 0.5)
        ddim_step(x_t, eps, 0.5, 0.9)
        self.assertTrue(np.array_equal(x_t, x_copy))
        self.assertTrue(np.array_equal(eps, eps_copy))


if __name__ == "__main__":
    unittest.main(verbosity=2)
