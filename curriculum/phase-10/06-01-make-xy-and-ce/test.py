import math
import unittest

from solution import cross_entropy, make_xy


class TestMakeXyAndCe(unittest.TestCase):
    def test_target_is_input_shifted_by_one(self):
        ids = [10, 11, 12, 13, 14]
        x, y = make_xy(ids, 3, 0)
        self.assertEqual(x, [10, 11, 12])
        self.assertEqual(y, [11, 12, 13])
        x, y = make_xy(ids, 3, 1)
        self.assertEqual(x, [11, 12, 13])
        self.assertEqual(y, [12, 13, 14])
        self.assertEqual(make_xy(ids, 1, 3), ([13], [14]))

    def test_every_target_follows_its_input(self):
        ids = list(range(100, 140))
        for i in range(0, 30):
            x, y = make_xy(ids, 8, i)
            self.assertEqual(len(x), 8)
            self.assertEqual(len(y), 8)
            self.assertEqual(y[:-1], x[1:])
            self.assertEqual(y[-1], ids[i + 8])

    def test_window_that_runs_off_the_end_raises(self):
        ids = [10, 11, 12, 13, 14]
        make_xy(ids, 4, 0)
        with self.assertRaises(ValueError):
            make_xy(ids, 3, 2)
        with self.assertRaises(ValueError):
            make_xy(ids, 5, 0)
        with self.assertRaises(ValueError):
            make_xy(ids, 2, -1)
        with self.assertRaises(ValueError):
            make_xy(ids, 0, 0)

    def test_uniform_logits_give_ln_v(self):
        for v in (2, 4, 27, 50):
            self.assertAlmostEqual(cross_entropy([[0.0] * v], [v - 1]), math.log(v), places=9)
            self.assertAlmostEqual(cross_entropy([[3.5] * v], [0]), math.log(v), places=9)

    def test_confident_correct_and_wrong_predictions(self):
        self.assertAlmostEqual(cross_entropy([[0.0, 100.0]], [1]), 0.0, places=9)
        self.assertAlmostEqual(cross_entropy([[0.0, 100.0]], [0]), 100.0, places=6)
        self.assertAlmostEqual(cross_entropy([[1.0, 2.0, 3.0]], [2]), math.log(math.e + math.e**2 + math.e**3) - 3.0, places=9)

    def test_large_logits_stay_finite(self):
        loss = cross_entropy([[1000.0, 1001.0]], [0])
        self.assertTrue(math.isfinite(loss))
        self.assertAlmostEqual(loss, math.log(1 + math.e), places=9)
        loss = cross_entropy([[-5000.0, -5000.0, -5000.0]], [1])
        self.assertAlmostEqual(loss, math.log(3), places=9)

    def test_mean_over_the_batch(self):
        loss = cross_entropy([[0.0, 0.0], [0.0, 100.0], [100.0, 0.0]], [0, 1, 1])
        self.assertAlmostEqual(loss, (math.log(2) + 0.0 + 100.0) / 3, places=6)

    def test_invalid_batches_raise(self):
        with self.assertRaises(ValueError):
            cross_entropy([], [])
        with self.assertRaises(ValueError):
            cross_entropy([[0.0, 1.0]], [0, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
