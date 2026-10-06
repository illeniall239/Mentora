import math
import unittest

from solution import chinchilla_tokens, train_flops


class TestFlopsAndChinchilla(unittest.TestCase):
    def test_six_flops_per_parameter_per_token(self):
        self.assertEqual(train_flops(1, 1), 6.0)
        self.assertIsInstance(train_flops(1, 1), float)
        self.assertTrue(math.isclose(train_flops(1000, 500), 3e6, rel_tol=1e-9))

    def test_chinchilla_run(self):
        self.assertTrue(math.isclose(train_flops(70e9, 1.4e12), 5.88e23, rel_tol=1e-9))

    def test_flops_scale_linearly_in_each_factor(self):
        base = train_flops(1e9, 2e10)
        self.assertTrue(math.isclose(train_flops(2e9, 2e10), 2 * base, rel_tol=1e-9))
        self.assertTrue(math.isclose(train_flops(1e9, 8e10), 4 * base, rel_tol=1e-9))
        self.assertTrue(math.isclose(train_flops(2e9, 8e10), 8 * base, rel_tol=1e-9))

    def test_seventy_billion_needs_one_point_four_trillion_tokens(self):
        self.assertTrue(math.isclose(chinchilla_tokens(70e9), 1.4e12, rel_tol=1e-9))
        self.assertTrue(math.isclose(chinchilla_tokens(1e9), 2e10, rel_tol=1e-9))
        self.assertIsInstance(chinchilla_tokens(5), float)
        self.assertEqual(chinchilla_tokens(5), 100.0)

    def test_compute_optimal_run_grows_quadratically(self):
        small = train_flops(1e9, chinchilla_tokens(1e9))
        self.assertTrue(math.isclose(small, 1.2e20, rel_tol=1e-9))
        self.assertTrue(math.isclose(train_flops(2e9, chinchilla_tokens(2e9)), 4 * small, rel_tol=1e-9))

    def test_non_positive_inputs_raise(self):
        for n, d in [(0, 100), (100, 0), (-1e9, 1e12), (1e9, -5)]:
            with self.assertRaises(ValueError):
                train_flops(n, d)
        for n in (0, -7e9):
            with self.assertRaises(ValueError):
                chinchilla_tokens(n)


if __name__ == "__main__":
    unittest.main(verbosity=2)
