import unittest

import numpy as np

from solution import recall_at_k

ASYM = np.array(
    [
        [0.9, 0.8, 0.7],
        [0.9, 0.5, 0.6],
        [0.2, 0.1, 0.3],
    ]
)


def expected(sim: np.ndarray, k: int) -> tuple[float, float]:
    """Reference ranking by stable descending sort, done the slow explicit way."""
    n = sim.shape[0]
    i2t = sum(1 for i in range(n) if i in list(np.argsort(-sim[i, :], kind="stable"))[:k])
    t2i = sum(1 for j in range(n) if j in list(np.argsort(-sim[:, j], kind="stable"))[:k])
    return i2t / n, t2i / n


class TestRecallAtK(unittest.TestCase):
    def test_perfect_matrix(self):
        self.assertEqual(recall_at_k(np.eye(4), 1), (1.0, 1.0))
        self.assertEqual(recall_at_k(np.eye(1), 1), (1.0, 1.0))

    def test_the_two_directions_differ(self):
        got = recall_at_k(ASYM, 1)
        self.assertAlmostEqual(got[0], 2 / 3, places=12)
        self.assertAlmostEqual(got[1], 1 / 3, places=12)
        got2 = recall_at_k(ASYM, 2)
        self.assertAlmostEqual(got2[0], 2 / 3, places=12)
        self.assertAlmostEqual(got2[1], 2 / 3, places=12)
        self.assertEqual(recall_at_k(ASYM, 3), (1.0, 1.0))

    def test_k_equals_n_is_always_perfect(self):
        rng = np.random.default_rng(0)
        for n in (2, 5, 9):
            sim = rng.normal(size=(n, n))
            self.assertEqual(recall_at_k(sim, n), (1.0, 1.0))

    def test_matches_a_stable_sort_on_random_matrices(self):
        rng = np.random.default_rng(1)
        for n in (4, 8, 16):
            sim = rng.normal(size=(n, n))
            for k in (1, 2, n // 2, n):
                want = expected(sim, k)
                got = recall_at_k(sim, k)
                self.assertAlmostEqual(got[0], want[0], places=12, msg=f"i2t n={n} k={k}")
                self.assertAlmostEqual(got[1], want[1], places=12, msg=f"t2i n={n} k={k}")

    def test_ties_go_to_the_lower_index(self):
        sim = np.array([[1.0, 1.0], [0.0, 1.0]])
        got = recall_at_k(sim, 1)
        self.assertAlmostEqual(got[0], 1.0, places=12)  # row 0 ties with itself at index 0
        self.assertAlmostEqual(got[1], 0.5, places=12)  # column 1 ties, image 0 ranks first

    def test_never_decreases_with_k(self):
        rng = np.random.default_rng(2)
        sim = rng.normal(size=(12, 12))
        i2t = [recall_at_k(sim, k)[0] for k in range(1, 13)]
        t2i = [recall_at_k(sim, k)[1] for k in range(1, 13)]
        self.assertEqual(i2t, sorted(i2t))
        self.assertEqual(t2i, sorted(t2i))
        self.assertLess(i2t[0], 1.0)

    def test_bad_inputs_raise(self):
        with self.assertRaises(ValueError):
            recall_at_k(np.zeros((2, 3)), 1)
        with self.assertRaises(ValueError):
            recall_at_k(np.zeros(4), 1)
        with self.assertRaises(ValueError):
            recall_at_k(np.eye(4), 0)
        with self.assertRaises(ValueError):
            recall_at_k(np.eye(4), 5)
        with self.assertRaises(ValueError):
            recall_at_k(np.zeros((0, 0)), 1)

    def test_input_is_not_modified(self):
        sim = ASYM.copy()
        recall_at_k(sim, 2)
        self.assertTrue(np.array_equal(sim, ASYM))


if __name__ == "__main__":
    unittest.main(verbosity=2)
