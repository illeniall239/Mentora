import unittest

import numpy as np

from solution import analogy, nearest

# Hand-made table: axis 0 = royalty, axis 1 = male (+) / female (-), axis 2 = young.
VOCAB = ["king", "queen", "man", "woman", "prince", "princess", "boy", "girl", "royal"]
TABLE = np.array(
    [
        [1.0, 1.0, 0.0],   # king
        [1.0, -1.0, 0.0],  # queen
        [0.0, 1.0, 0.0],   # man
        [0.0, -1.0, 0.0],  # woman
        [1.0, 1.0, 1.0],   # prince
        [1.0, -1.0, 1.0],  # princess
        [0.0, 1.0, 1.0],   # boy
        [0.0, -1.0, 1.0],  # girl
        [3.0, 0.5, 0.0],   # royal: a long vector, far away in Euclidean distance
    ]
)


class TestNearestAndAnalogy(unittest.TestCase):
    def test_nearest_by_direction_not_distance(self):
        # Euclidean distance would rank king (0.9 away) before royal (2.1 away).
        self.assertEqual(nearest(np.array([1.0, 0.1, 0.0]), TABLE, 2), [8, 0])

    def test_nearest_ignores_row_length(self):
        # The raw dot product would rank prince (1.15) and king (1.1) above man (1.0).
        self.assertEqual(nearest(np.array([0.1, 1.0, 0.05]), TABLE, 3), [2, 0, 6])

    def test_nearest_ignores_query_scale(self):
        q = np.array([0.3, -0.7, 0.2])
        self.assertEqual(nearest(q, TABLE, 4), nearest(5.0 * q, TABLE, 4))

    def test_nearest_full_ranking_is_a_permutation(self):
        rng = np.random.default_rng(0)
        table = rng.normal(size=(12, 4))
        got = nearest(rng.normal(size=4), table, 12)
        self.assertEqual(sorted(got), list(range(12)))
        self.assertEqual(len(nearest(rng.normal(size=4), table, 1)), 1)

    def test_nearest_matches_brute_force_cosine(self):
        rng = np.random.default_rng(1)
        table = rng.normal(size=(20, 6)) * rng.uniform(0.1, 5.0, size=(20, 1))
        q = rng.normal(size=6)
        cos = [float(q @ r / (np.linalg.norm(q) * np.linalg.norm(r))) for r in table]
        want = sorted(range(20), key=lambda i: -cos[i])[:5]
        self.assertEqual(nearest(q, table, 5), want)

    def test_classic_analogies(self):
        self.assertEqual(analogy("man", "king", "woman", TABLE, VOCAB), "queen")
        self.assertEqual(analogy("woman", "queen", "man", TABLE, VOCAB), "king")
        self.assertEqual(analogy("king", "prince", "queen", TABLE, VOCAB), "princess")
        self.assertEqual(analogy("boy", "man", "girl", TABLE, VOCAB), "woman")

    def test_analogy_never_returns_an_input_word(self):
        # king - king + queen is exactly queen's vector; queen must be skipped.
        self.assertEqual(analogy("king", "king", "queen", TABLE, VOCAB), "princess")
        self.assertEqual(analogy("man", "man", "boy", TABLE, VOCAB), "prince")

    def test_analogy_unknown_word_raises(self):
        with self.assertRaises(ValueError):
            analogy("man", "king", "duke", TABLE, VOCAB)
        with self.assertRaises(ValueError):
            analogy("Man", "king", "woman", TABLE, VOCAB)


if __name__ == "__main__":
    unittest.main(verbosity=2)
