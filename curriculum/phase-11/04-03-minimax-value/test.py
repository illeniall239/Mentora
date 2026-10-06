import math
import unittest

import numpy as np

from solution import minimax_value, optimal_d

NEG_LOG4 = -math.log(4.0)


def random_dist(rng, k: int) -> np.ndarray:
    p = rng.uniform(0.0, 1.0, size=k)
    return p / p.sum()


class TestOptimalD(unittest.TestCase):
    def test_half_everywhere_when_the_generator_matches(self):
        p = np.array([0.25, 0.25, 0.5])
        self.assertTrue(np.allclose(optimal_d(p, p), 0.5, atol=1e-12))
        rng = np.random.default_rng(0)
        q = random_dist(rng, 20)
        self.assertTrue(np.allclose(optimal_d(q, q), 0.5, atol=1e-12))

    def test_disjoint_supports_give_ones_and_zeros(self):
        d = optimal_d(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        self.assertTrue(np.allclose(d, [1.0, 0.0], atol=1e-12))

    def test_zero_over_zero_is_a_half(self):
        p = np.array([0.5, 0.5, 0.0])
        self.assertTrue(np.allclose(optimal_d(p, p), [0.5, 0.5, 0.5], atol=1e-12))
        d = optimal_d(np.array([0.5, 0.5, 0.0]), np.array([1.0, 0.0, 0.0]))
        self.assertTrue(np.allclose(d, [1.0 / 3.0, 1.0, 0.5], atol=1e-12))

    def test_matches_the_ratio_and_stays_in_range(self):
        rng = np.random.default_rng(1)
        for k in (2, 7, 64):
            a, b = random_dist(rng, k), random_dist(rng, k)
            d = optimal_d(a, b)
            self.assertEqual(d.shape, (k,))
            self.assertTrue(np.allclose(d, a / (a + b), atol=1e-12))
            self.assertTrue(bool(np.all((d >= 0.0) & (d <= 1.0))))


class TestMinimaxValue(unittest.TestCase):
    def test_minus_log_four_when_the_generator_matches(self):
        rng = np.random.default_rng(2)
        for k in (2, 5, 32):
            p = random_dist(rng, k)
            self.assertAlmostEqual(minimax_value(p, p), NEG_LOG4, places=9)
        self.assertAlmostEqual(minimax_value(np.array([1.0, 0.0]), np.array([1.0, 0.0])), NEG_LOG4, places=9)

    def test_zero_for_disjoint_supports(self):
        self.assertAlmostEqual(minimax_value(np.array([1.0, 0.0]), np.array([0.0, 1.0])), 0.0, places=12)
        a = np.array([0.5, 0.5, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.25, 0.75])
        self.assertAlmostEqual(minimax_value(a, b), 0.0, places=12)

    def test_hand_computed_value(self):
        a, b = np.array([0.5, 0.5]), np.array([1.0, 0.0])
        want = 0.5 * math.log(1.0 / 3.0) + 1.0 * math.log(2.0 / 3.0)
        self.assertAlmostEqual(minimax_value(a, b), want, places=12)
        self.assertAlmostEqual(want, -0.9547712524422192, places=12)

    def test_minus_log_four_is_the_floor(self):
        rng = np.random.default_rng(3)
        for _ in range(50):
            k = int(rng.integers(2, 12))
            a, b = random_dist(rng, k), random_dist(rng, k)
            v = minimax_value(a, b)
            self.assertGreaterEqual(v, NEG_LOG4 - 1e-9)
            self.assertLessEqual(v, 1e-9)

    def test_rises_as_the_generator_drifts_away(self):
        base = np.array([0.5, 0.5])
        values = [minimax_value(base, np.array([0.5 + s, 0.5 - s])) for s in (0.0, 0.1, 0.25, 0.4, 0.5)]
        self.assertEqual(values, sorted(values))
        self.assertAlmostEqual(values[0], NEG_LOG4, places=9)
        self.assertGreater(values[-1], NEG_LOG4 + 0.3)

    def test_zero_weight_terms_do_not_poison_the_sum(self):
        a = np.array([0.5, 0.5, 0.0])
        b = np.array([0.4, 0.6, 0.0])
        value = minimax_value(a, b)
        self.assertTrue(math.isfinite(value))
        two_point = minimax_value(a[:2] / a[:2].sum(), b[:2] / b[:2].sum())
        self.assertAlmostEqual(value, two_point, places=12)

    def test_bad_inputs_raise(self):
        good = np.array([0.5, 0.5])
        for bad in (np.array([0.5, 0.6]), np.array([-0.5, 1.5]), np.array([]), np.array([[0.5, 0.5]])):
            with self.assertRaises(ValueError):
                minimax_value(bad, good)
            with self.assertRaises(ValueError):
                optimal_d(good, bad)
        with self.assertRaises(ValueError):
            minimax_value(np.array([1.0, 0.0]), np.array([1.0]))

    def test_inputs_are_not_modified(self):
        a, b = np.array([0.25, 0.75]), np.array([0.5, 0.5])
        a_copy, b_copy = a.copy(), b.copy()
        optimal_d(a, b)
        minimax_value(a, b)
        self.assertTrue(np.array_equal(a, a_copy))
        self.assertTrue(np.array_equal(b, b_copy))


if __name__ == "__main__":
    unittest.main(verbosity=2)
