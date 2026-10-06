import math
import random
import unittest

from solution import pass_at_k, pass_at_k_from_results


class TestPassAtK(unittest.TestCase):
    def test_hand_values(self):
        self.assertAlmostEqual(pass_at_k(10, 1, 1), 0.1, places=9)
        self.assertAlmostEqual(pass_at_k(10, 1, 5), 0.5, places=9)
        self.assertAlmostEqual(pass_at_k(10, 5, 2), 1 - 10 / 45, places=9)
        self.assertAlmostEqual(pass_at_k(100, 10, 10), 1 - math.comb(90, 10) / math.comb(100, 10), places=9)

    def test_pass_at_1_is_the_success_rate(self):
        for n, c in [(10, 1), (10, 7), (8, 8), (200, 3)]:
            self.assertAlmostEqual(pass_at_k(n, c, 1), c / n, places=9)

    def test_edge_cases(self):
        self.assertEqual(pass_at_k(5, 0, 3), 0.0)
        self.assertEqual(pass_at_k(5, 0, 5), 0.0)
        self.assertEqual(pass_at_k(10, 1, 10), 1.0)
        self.assertEqual(pass_at_k(5, 5, 1), 1.0)
        self.assertEqual(pass_at_k(10, 4, 7), 1.0)
        self.assertEqual(pass_at_k(1, 1, 1), 1.0)

    def test_not_the_first_k_samples_estimator(self):
        # One success in ten samples: the estimate is 0.5 at k = 5, neither 0.1 nor 1.0.
        got = pass_at_k(10, 1, 5)
        self.assertNotAlmostEqual(got, 0.1, places=3)
        self.assertNotAlmostEqual(got, 1.0, places=3)
        self.assertAlmostEqual(got, 0.5, places=9)

    def test_monotone_in_k_and_in_c(self):
        in_k = [pass_at_k(20, 3, k) for k in range(1, 21)]
        self.assertEqual(in_k, sorted(in_k))
        self.assertAlmostEqual(in_k[-1], 1.0, places=9)
        in_c = [pass_at_k(20, c, 5) for c in range(0, 21)]
        self.assertEqual(in_c, sorted(in_c))
        self.assertEqual(in_c[0], 0.0)

    def test_from_results_ignores_order(self):
        results = [True] + [False] * 9
        self.assertAlmostEqual(pass_at_k_from_results(results, 5), 0.5, places=9)
        self.assertAlmostEqual(pass_at_k_from_results([False] * 9 + [True], 5), 0.5, places=9)
        shuffled = list(results)
        random.Random(0).shuffle(shuffled)
        self.assertAlmostEqual(pass_at_k_from_results(shuffled, 5), 0.5, places=9)
        self.assertAlmostEqual(pass_at_k_from_results([True, False, True, False], 2), pass_at_k(4, 2, 2), places=9)

    def test_validation(self):
        with self.assertRaises(ValueError):
            pass_at_k(5, 2, 6)
        with self.assertRaises(ValueError):
            pass_at_k(5, 6, 2)
        with self.assertRaises(ValueError):
            pass_at_k(5, -1, 2)
        with self.assertRaises(ValueError):
            pass_at_k(5, 2, 0)
        with self.assertRaises(ValueError):
            pass_at_k(0, 0, 1)
        with self.assertRaises(ValueError):
            pass_at_k_from_results([], 1)
        with self.assertRaises(ValueError):
            pass_at_k_from_results([True, False], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
